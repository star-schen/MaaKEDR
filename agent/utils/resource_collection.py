"""Sequential resource collection with explicit stage/team/count verification.

All coordinates are in MaaFramework's 1280x720 controller space. No ordinary
battle, stamina purchase, or blind retry of a submitted battle is allowed.
"""
from __future__ import annotations

import json
import re
import struct
import time
import zlib
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from maa.custom_action import CustomAction
from maa.pipeline import JOCR, JRecognitionType, JTemplateMatch
from utils.logger import logger
from utils.params import parse_params

CATEGORIES = (("特别军费行动", 5), ("作战体能训练", 4), ("兵种能力评级", 4), ("载具对抗演练", 5))
CATEGORY_KEYS = ("funds", "physical", "rating", "vehicle")
PREFIX = "IntegratedResources"


class NavigationError(RuntimeError):
    pass


class NoStamina(RuntimeError):
    pass


class Cancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class StagePlan:
    category: int
    index: int
    team: int

    @property
    def name(self):
        return f"{self.category}-{self.index}"


def integer(value, low, high, label):
    if isinstance(value, bool) or not re.fullmatch(r"\d+", str(value)):
        raise ValueError(f"{label} 必须为整数")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{label} 必须在 {low}–{high} 之间")
    return value


def load_plan(context, params):
    team = integer(params.get("team"), 1, 5, "指定队伍")
    plan = []
    for category, (_, size) in enumerate(CATEGORIES, 1):
        key = CATEGORY_KEYS[category-1]
        node = context.get_node_data(f"{PREFIX}.Category.{key}")
        if not node or type(node.get("enabled")) is not bool:
            raise ValueError("刷取大类配置不完整，请重新选择任务选项")
        if node["enabled"]:
            plan.extend(StagePlan(category, index, team) for index in range(1, size+1))
    return plan


def team_selected(image, team):
    """Each selected team has a yellow marker at the left edge of its own row."""
    y = 156 + (team - 1) * 73
    crop = image[y-18:y+18, 8:20, :3].astype(np.int16)
    b, g, r = crop[:, :, 0], crop[:, :, 1], crop[:, :, 2]
    return float(np.mean((r > 140) & (g > 130) & (b < 125) & (g > r * .65))) > .3


class ResourceCollector:
    def __init__(self, context):
        self.context = context
        self.controller = context.tasker.controller
        self.results = []
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.report_folder = Path("logs/integrated_resources")
        self.report = {"started": datetime.now().isoformat(timespec="seconds"),
                       "status": "running", "stages": self.results, "navigation": []}
        self.image = None
        self.frame = 0
        self.recognitions = deque(maxlen=100)
        self.map_side = None
        self.current_stage = None

    def record_recognition(self, kind, query, roi, detail):
        results = []
        if detail:
            for result in detail.all_results[:8]:
                item = {"box": list(result.box)}
                if hasattr(result, "text"):
                    item["text"] = result.text
                if hasattr(result, "score"):
                    item["score"] = float(result.score)
                results.append(item)
        self.recognitions.append({
            "frame": self.frame, "kind": kind, "query": query, "roi": list(roi),
            "hit": bool(detail and detail.hit), "results": results})

    def navigation(self, event, **fields):
        self.report["navigation"].append({
            "time": datetime.now().isoformat(timespec="milliseconds"),
            "event": event, **fields})

    def save_evidence(self, reason):
        """Preserve the decision frame and recent hits/misses without another capture."""
        folder = self.report_folder / self.run_id
        label = self.current_stage.name if self.current_stage else "navigation"
        stem = f"{label}_{len(self.report.get('evidence', []))+1}"
        evidence = {"stage": label, "reason": reason, "frame": self.frame}
        try:
            folder.mkdir(parents=True, exist_ok=True)
            if self.image is not None:
                # A small RGB PNG writer avoids introducing another runtime dependency.
                height, width = self.image.shape[:2]
                pixels = self.image[:, :, :3][:, :, ::-1]
                raw = b"".join(b"\0" + row.tobytes() for row in pixels)
                def chunk(kind, data):
                    return (struct.pack(">I", len(data)) + kind + data
                            + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))
                png = (b"\x89PNG\r\n\x1a\n"
                       + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                       + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
                path = folder / f"{stem}.png"
                path.write_bytes(png)
                evidence["screenshot"] = str(path.resolve())
            path = folder / f"{stem}.json"
            path.write_text(json.dumps({**evidence, "recognitions": list(self.recognitions)},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
            evidence["recognitions"] = str(path.resolve())
        except Exception as error:
            evidence["save_error"] = str(error)
            logger.warning("[统合资源刷取] 保存反馈失败：{}", error)
        self.report.setdefault("evidence", []).append(evidence)
        if self.current_stage and self.results:
            self.results[-1].setdefault("evidence", []).append(evidence)

    def check_cancel(self):
        if self.context.tasker.stopping:
            raise Cancelled("用户停止任务")

    def pause(self, seconds=.5):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.check_cancel()
            time.sleep(min(.1, max(0, end-time.monotonic())))

    def capture(self):
        self.check_cancel()
        if not self.controller.post_screencap().wait().succeeded:
            raise NavigationError("ADB 截图失败")
        self.image = self.controller.cached_image
        if self.image is None or self.image.shape[:2] != (720, 1280):
            raise NavigationError("请将控制器设为短边720，模拟器使用16:9横屏")
        self.frame += 1
        return self.image

    def tap(self, x, y, delay=.7):
        self.check_cancel()
        if not self.controller.post_click(int(x), int(y)).wait().succeeded:
            raise NavigationError("ADB 点击失败")
        self.pause(delay)

    def click(self, detail, delay=.7):
        x, y, w, h = detail.box
        self.tap(x+w//2, y+h//2, delay)

    def swipe(self, right):
        self.check_cancel()
        start, end = ((250, 900) if right else (1100, 150))
        if not self.controller.post_swipe(start, 400, end, 400, 500).wait().succeeded:
            raise NavigationError("ADB 滑动失败")
        self.pause(.6)

    def ocr(self, expected, roi=(0, 0, 1280, 720)):
        detail = self.context.run_recognition_direct(
            JRecognitionType.OCR, JOCR(expected=[expected], roi=roi), self.image)
        self.record_recognition("OCR", expected, roi, detail)
        return detail if detail and detail.hit and detail.box else None

    def template(self, name, roi=(0, 0, 1280, 720), threshold=.8):
        detail = self.context.run_recognition_direct(
            JRecognitionType.TemplateMatch,
            JTemplateMatch(template=[name], roi=roi, threshold=[threshold]), self.image)
        self.record_recognition("TemplateMatch", {"template": name, "threshold": threshold}, roi, detail)
        return detail if detail and detail.hit and detail.box else None

    def wait_for(self, detect, seconds=12):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            self.capture()
            result = detect()
            if result:
                return result
            self.pause(.35)
        return None

    def log(self, message):
        logger.info("[统合资源刷取] {}", message)
        if not self.context.tasker.stopping:
            self.context.run_task(f"{PREFIX}.Message", {f"{PREFIX}.Message": {
                "recognition": "DirectHit", "action": "DoNothing", "next": [],
                "pre_delay": 0, "post_delay": 0,
                "focus": {"Node.Action.Starting": message}}})

    def stamina_popup(self):
        return (self.template("farm_resources/no_stamina.png", (450, 115, 360, 140))
                or self.ocr("(体力|指挥点数|指挥点).*(不足|补充|恢复)|购买体力", (280, 110, 760, 480)))

    def ready(self):
        return (self.ocr("^准备快速战斗$", (785, 628, 220, 42))
                or self.template("farm_resources/prepare_battle.png", (778, 613, 228, 67), .85))

    def stage_dialog(self):
        return self.template("farm_resources/exit_stage_confirm.png", (1060, 20, 110, 85))

    def home(self):
        return self.template("main_option.png", (1190, 0, 90, 100))

    def go_home(self, reason="start"):
        self.navigation("go_home", reason=reason)
        for _ in range(10):
            self.capture()
            if self.home():
                return
            if self.stamina_popup():
                raise NoStamina("体力不足，结束整个统合任务")
            else:
                close = self.stage_dialog()
                if close:
                    self.click(close)
                    continue
                button = self.template("return_main.png", (180, 20, 120, 65))
                if not button:
                    self.pause(.6)
                    continue
                self.click(button, 1)
        raise NavigationError("无法返回主页，已停止以避免误点")

    def collection_list(self):
        return (self.ocr("^资源收集$", (510, 630, 240, 85))
                and self.ocr("|".join(label for label, _ in CATEGORIES), (0, 490, 1280, 135)))

    def select_at_position(self, label, x, y, selected, on_selection_page):
        """Bounded retries for idempotent tab/team selection, never battle submission."""
        def checked_selected():
            if self.stamina_popup():
                raise NoStamina("体力不足，结束整个统合任务")
            return selected()

        def page_ready():
            return checked_selected() or on_selection_page()

        for attempt in range(1, 4):
            if not self.wait_for(page_ready, 4):
                raise NavigationError(f"{label}：无法确认所在页面，停止点击")
            # Text can appear before the page accepts input. Let entry animation settle.
            self.pause(.8)
            self.capture()
            if checked_selected():
                return
            if not on_selection_page():
                raise NavigationError(f"{label}：等待后页面状态改变，停止点击")
            self.navigation("fixed_selection_click", label=label, attempt=attempt, x=x, y=y)
            if attempt > 1:
                self.log(f"{label}尚未生效，确认仍在选择页面，补点第{attempt-1}次")
            self.tap(x, y, 1)
            if self.wait_for(checked_selected, 3):
                return
        raise NavigationError(f"{label}：固定位置点击3次后仍未确认成功，已停止")

    def category_map(self, category):
        return self.ocr(CATEGORIES[category-1][0], (530, 80, 360, 75))

    def open_collection(self):
        button = self.template("farm_resources/battle_entry.png", (1020, 550, 170, 130))
        if not button:
            raise NavigationError("未找到出击入口")
        self.click(button)
        def on_tabs():
            return (self.ocr("^资源收集$", (510, 630, 240, 85))
                    and self.ocr("^主线任务$", (340, 630, 170, 85)))
        self.select_at_position("切换资源收集", 630, 665, self.collection_list, on_tabs)
        self.navigation("collection_list", source="home")

    def open_category(self, category):
        label = CATEGORIES[category-1][0]
        self.map_side = None
        # The fourth card initially extends beyond the right edge.
        for _ in range(3):
            button = self.wait_for(lambda: self.ocr(label, (0, 490, 1280, 135)), 3)
            if button:
                self.click(button)
                if self.wait_for(lambda: self.category_map(category), 8):
                    self.navigation("enter_category", category=category)
                    return
                raise NavigationError(f"无法确认 {label} 关卡列表")
            self.swipe(category <= 2)
        raise NavigationError(f"未找到资源分类 {label}")

    def return_to_category(self, category):
        end = time.monotonic() + 10
        while time.monotonic() < end:
            self.capture()
            if self.stamina_popup():
                raise NoStamina("体力不足，结束整个统合任务")
            close = self.stage_dialog()
            if close:
                self.click(close)
            elif self.category_map(category):
                self.navigation("category_map", category=category,
                                after_stage=self.current_stage.name if self.current_stage else None)
                return
            else:
                self.pause(.35)
        raise NavigationError(f"无法返回 {CATEGORIES[category-1][0]} 地图，已停止")

    def return_to_collection(self, category):
        self.return_to_category(category)
        # This back arrow is used only after the current category map was recognized.
        self.tap(35, 52)
        if not self.wait_for(self.collection_list, 8):
            raise NavigationError("切换大类时无法确认资源收集列表，已停止")
        self.navigation("collection_list", source="category", category=category)

    def open_stage(self, stage):
        # Reset only on entry or when moving to the map's other end.
        right = stage.index <= 3
        if self.map_side != right:
            for _ in range(2):
                self.swipe(right)
            self.map_side = right
        x, y = ((170+(stage.index-1)*489, 482 if stage.index % 2 else 532)
                if stage.index <= 3 else
                ((317 if stage.index == 4 else 807), 530 if stage.index == 4 else 483))
        if stage.index == 4 and CATEGORIES[stage.category-1][1] == 4:
            x = 806
        self.capture()
        if self.template("farm_resources/lock_icon.png", (x-15, y-15, 120, 90), .7):
            return "locked"
        number = self.ocr(rf"^{stage.category}\s*[-一—–.]?\s*{stage.index}$", (x-25, y-15, 150, 95))
        if not number:
            # Never click the coordinates unless the requested stage was read.
            raise NavigationError(f"未识别到关卡 {stage.name}，已停止")
        self.click(number)
        if not self.wait_for(self.stage_dialog, 8):
            raise NavigationError(f"点击 {stage.name} 后未出现关卡详情，原因不明，请反馈后处理")
        if not self.wait_for(lambda: self.ocr(
                rf"^{stage.category}\s*[-一—–.]?\s*{stage.index}$", (125, 25, 150, 75)), 4):
            raise NavigationError(f"关卡标题与 {stage.name} 不符")
        return self.wait_quick_battle(stage)

    def wait_quick_battle(self, stage):
        end = time.monotonic() + 12
        toggled = False
        unavailable_frames = 0
        while time.monotonic() < end:
            self.capture()
            if self.stamina_popup():
                raise NoStamina("体力不足，结束整个统合任务")
            # Already-on must win: the game's toggle remembers the previous stage.
            if self.ready():
                self.results[-1]["quick_mode"] = "enabled"
                return "ready"
            enabled = self.ocr("^关闭$", (815, 500, 105, 70))
            if enabled:
                unavailable_frames = 0
                self.pause(.35)
                continue
            unavailable = self.ocr(
                r"快速战斗.{0,8}(未解锁|不可用|无法使用|无法进行)|"
                r"(未解锁|不可|无法|不支持|不能).{0,6}快速战斗|"
                r"(通关|三星|3星).{0,10}(解锁|开启)快速战斗", (650, 400, 490, 290))
            unavailable_frames = unavailable_frames + 1 if unavailable else 0
            if unavailable_frames >= 2:
                return "no_quick_battle"
            if not unavailable and not toggled:
                toggle = (self.ocr("^打开$", (900, 500, 110, 70))
                          or self.template("farm_resources/quick_battle.png", (895, 495, 125, 80), .88))
                if toggle:
                    self.click(toggle)
                    toggled = True
                    self.results[-1]["quick_mode"] = "toggle_clicked"
                    continue
            self.pause(.35)
        raise NavigationError(f"{stage.name}：12秒内无法确认快速战斗状态，未将识别失败当作不可用")

    def read_count(self):
        detail = self.ocr("^[1-6]$", (899, 436, 40, 53))
        if detail:
            for result in detail.filtered_results:
                if re.fullmatch("[1-6]", result.text.strip()):
                    return int(result.text.strip())
        return None

    def set_count_one(self):
        # Reset the game's remembered count before every stage.
        minimum = self.wait_for(lambda: self.ready() and self.ocr("^最小$", (710, 500, 100, 75)), 6)
        if not minimum:
            raise NavigationError("快速战斗设置界面不完整")
        self.click(minimum, .4)
        if not self.wait_for(lambda: self.read_count() == 1, 4):
            raise NavigationError("无法确认快速战斗次数为1")
        self.capture()
        if not self.ready() or self.read_count() != 1:
            raise NavigationError("快速战斗次数核验失败")
        self.results[-1]["count_confirmed"] = 1

    def prepare_team(self, team):
        button = self.wait_for(lambda: self.ready() if self.read_count() == 1 else None, 4)
        if not button:
            raise NavigationError("准备前无法确认快速战斗次数为1")
        self.click(button)
        def detect():
            if self.stamina_popup():
                raise NoStamina("体力不足，结束整个统合任务")
            return self.ocr("^开始战斗$", (1000, 635, 260, 80))
        if not self.wait_for(detect, 12):
            raise NavigationError("准备快速战斗后未出现选队界面")
        def selected():
            return (team_selected(self.image, team)
                    and self.ocr("^编辑$", (250, 128+(team-1)*73, 80, 60))
                    and self.ocr("^开始战斗$", (1000, 635, 260, 80)))
        def on_teams():
            return (self.ocr("^开始战斗$", (1000, 635, 260, 80))
                    and any(team_selected(self.image, slot) for slot in range(1, 6)))
        self.select_at_position(f"选择队伍{team}", 48, 156+(team-1)*73, selected, on_teams)
        self.results[-1]["team_confirmed"] = team

    def fight(self):
        self.capture()
        if self.stamina_popup():
            raise NoStamina("体力不足，结束整个统合任务")
        button = self.ocr("^开始战斗$", (1000, 635, 260, 80))
        if not button:
            raise NavigationError("开始按钮不可见")
        # This is the only battle submission. A timeout must not resubmit it.
        self.results[-1]["submission_attempted"] = True
        self.click(button, 1.5)
        saw_result = False
        end = time.monotonic()+90
        while time.monotonic() < end:
            self.capture()
            if self.stamina_popup():
                raise NoStamina("体力不足，结束整个统合任务")
            victory = self.template("farm_resources/battle_victory.png", (0, 0, 350, 160))
            items = self.template("item_obtained_dialog.png", (430, 170, 420, 160))
            if victory or items:
                saw_result = True
                self.click(victory or items, 1)
            elif saw_result and (self.stage_dialog() or self.ocr("BATTLE FIELD", (540, 85, 310, 75))):
                return
            else:
                self.pause(.5)
        raise NavigationError("已提交战斗但未确认结算，已停止；请核对游戏记录，避免重复刷取")

    def run_stage(self, stage):
        self.current_stage = stage
        self.recognitions.clear()
        row = {**asdict(stage), "stage": stage.name, "completed": 0, "status": "running"}
        self.results.append(row)
        self.log(f"{CATEGORIES[stage.category-1][0]} {stage.name}：队伍{stage.team}，快速战斗1次")
        state = self.open_stage(stage)
        if state != "ready":
            row["status"] = state
            row["reason"] = "关卡未解锁" if state == "locked" else "该关卡无法快速战斗"
            self.save_evidence(row["reason"])
            self.log(f"{stage.name}：{row['reason']}，跳过")
            return
        self.set_count_one()
        self.prepare_team(stage.team)
        self.fight()
        row["completed"] = 1
        row["status"] = "completed"
        self.log(f"{stage.name}：已完成1次快速战斗")

    def summary(self):
        done = sum(row["completed"] for row in self.results)
        skipped = [row for row in self.results if row["status"] in ("locked", "no_quick_battle")]
        self.log(f"本次完成{done}个小关卡，跳过{len(skipped)}个，尚未处理{len(self.report['plan'])-len(self.results)}个")
        if skipped:
            self.log("跳过明细："+"；".join(f"{row['stage']} {row['reason']}" for row in skipped))

    def run(self, plan):
        self.report["plan"] = [asdict(item) for item in plan]
        try:
            if not plan:
                self.report["status"] = "nothing_selected"
                self.log("未勾选刷取大类，本次不执行任何关卡")
                return True
            self.go_home("start")
            self.open_collection()
            category = None
            for stage in plan:
                self.check_cancel()
                if stage.category != category:
                    self.current_stage = None
                    if category is not None:
                        self.return_to_collection(category)
                    self.open_category(stage.category)
                    category = stage.category
                self.run_stage(stage)
                self.return_to_category(category)
            self.current_stage = None
            self.go_home("finish")
            self.report["status"] = "completed"
            self.log("已执行完全部勾选大类")
            self.summary()
            return True
        except NoStamina as error:
            self.report["status"] = "no_stamina"
            self.report["reason"] = str(error)
            if self.results and self.results[-1]["status"] == "running":
                self.results[-1]["status"] = "no_stamina"
                self.results[-1]["reason"] = str(error)
            self.save_evidence(str(error))
            self.log(str(error))
            self.summary()
            return True
        except Cancelled:
            self.report["status"] = "cancelled"
            return False
        except Exception as error:
            self.report["status"] = "error"
            self.report["error"] = str(error)
            if self.results and self.results[-1]["status"] == "running":
                self.results[-1]["status"] = "error"
                self.results[-1]["reason"] = str(error)
            self.save_evidence(str(error))
            self.log(f"任务已停止：{error}。请将日志反馈后再继续。")
            self.summary()
            return False
        finally:
            self.report["finished"] = datetime.now().isoformat(timespec="seconds")
            self.report_folder.mkdir(parents=True, exist_ok=True)
            path = self.report_folder / f"{self.run_id}.json"
            path.write_text(json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("[统合资源刷取] 执行报告：{}", path)


class IntegratedResourceFarm(CustomAction):
    def run(self, context, argv):
        try:
            params = parse_params(argv.custom_action_param)
            plan = load_plan(context, params)
        except (ValueError, KeyError, TypeError) as error:
            logger.error("统合资源刷取配置错误：{}", error)
            return False
        return ResourceCollector(context).run(plan)
