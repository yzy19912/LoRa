"""Rewrite data/train.jsonl into GOAP-formatted training data.

For every question the response is restructured into a goal-oriented
action-planning trace before the final CONCEPT/ANSWER:
  GOAL
  WORLD STATE
  ACTIONS  (each with PRECONDITIONS / EFFECTS)
  PLAN     (chosen action sequence)
  CONCEPT
  ANSWER
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_JSONL = ROOT / "data" / "train.jsonl"
DST_JSONL = ROOT / "data" / "train_goap.jsonl"


# ----------------------------------------------------------
# action library
# ----------------------------------------------------------


def a(name, pre, eff):
    return {"name": name, "pre": pre, "eff": eff}


ACT_DEFINE = a(
    "界定答题对象",
    "问题包含明确的主体、时期或地点",
    "明确「{concept}」为答题对象。",
)

ACT_RETRIEVE = a(
    "检索史实",
    "答题对象已界定",
    "取得关于「{concept}」的核心史实材料。",
)

ACT_ENUMERATE = a(
    "枚举整合",
    "核心史实材料已取得",
    "将「{concept}」相关的并列项完整列出，不遗漏。",
)

ACT_VERIFY = a(
    "核对史实",
    "核心史实材料已取得",
    "{hint}",
)

ACT_CAUSE = a(
    "追溯因果",
    "核心史实材料已取得",
    "{hint}",
)

ACT_OUTPUT = a(
    "组织输出",
    "史实已核对无误",
    "生成 CONCEPT 与 ANSWER，达成 GOAL。",
)


# ----------------------------------------------------------
# per-row profiles: (kind, world-state hint, verify/cause eff hint)
#   kind: default | causal | enum
# ----------------------------------------------------------

PROFILES = [
    (
        "default",
        "问题指向井盐生产的起源时段，需从史料中定位有文字可考的最早阶段。",
        "确认起源时段为东汉、以汉章帝时富世盐井开凿为起点，时间点与史料记载一致。",
    ),
    (
        "default",
        "对象为传说人物，需区分「盐业始祖」与「盐井之神」两种身份。",
        "确认所指人物为僚人领袖梅泽，其在井旁发现咸泉并凿井取盐的传说与之吻合。",
    ),
    (
        "causal",
        "对象为富世盐井得名，需结合西晋太康元年的产盐状况解释。",
        "因产盐量大、经济效益显著而称「富世」，得名与西晋太康元年的记载对应。",
    ),
    (
        "default",
        "需从《旧唐书·地理志》原文中检索井深数据。",
        "确认深度「约77.5米」与《旧唐书·地理志》所载一致。",
    ),
    (
        "default",
        "对象为大公井的开凿者，涉及「僚人」参与早期开凿的记载。",
        "确认大公井由僚人开凿，且曾与富世井齐名。",
    ),
    (
        "causal",
        "对象为冶官县得名，需结合当地盐铁兼具的资源状况解释。",
        "因「有盐有铁」而得冶官之名，得名逻辑与史籍解释一致。",
    ),
    (
        "default",
        "需在自贡市区建制的沿革中定位最早的行政建制单位。",
        "确认北周时期所设的公井镇为今自贡市区最早的行政建制单位。",
    ),
    (
        "default",
        "对象为北周天和二年在富世盐井附近设置的行政单位。",
        "确认北周天和二年（567年）设置雒原郡与富世县。",
    ),
    (
        "default",
        "对象为唐武德元年荣州初治所在。",
        "确认荣州最初治于公井镇，即今贡井区一带。",
    ),
    ("default", "对象为唐代县名「婆日」的词源。", "确认「婆日」为僚语「铁山」之意。"),
    ("default", "对象为唐代县名「至如」的词源。", "确认「至如」为僚语「盐」之意。"),
    (
        "causal",
        "对象为荣州州治的迁移，需交代迁治的时间与走向。",
        "唐贞观六年从公井县迁往旭川县，方向为自贡市区方向迁向荣县方向，与史料吻合。",
    ),
    (
        "causal",
        "对象为富世县改名的原因，涉及避讳制度。",
        "为避唐太宗李世民名讳改名富义县，改名逻辑符合唐代避讳惯例。",
    ),
    (
        "causal",
        "对象为富义监改名的原因，涉及避讳制度。",
        "为避宋太宗赵光义名讳改名富顺监，改名逻辑符合宋代避讳惯例。",
    ),
    (
        "causal",
        "对象为旭川县改名的原因，涉及避讳制度。",
        "为避宋神宗赵顼名讳改名荣德县，改名逻辑符合宋代避讳惯例。",
    ),
    (
        "causal",
        "对象为卓筒井对采卤方式的影响。",
        "卓筒井使采卤从小口深井取代大口浅井，推动井盐业扩张，因果链完整。",
    ),
    (
        "default",
        "对象为自贡历史上第一次升格为「府」时的府名。",
        "确认南宋绍定六年荣州因潜邸升格为绍熙府。",
    ),
    (
        "default",
        "对象为南宋绍熙府迁治地点。",
        "确认1236年迁至鸿鹤镇，即今自流井区鸿鹤坝一带。",
    ),
    (
        "default",
        "对象为南宋时期富顺抗元的重要城池。",
        "确认富顺监府迁至虎头山筑城抗元的记载。",
    ),
    ("default", "对象为虎头城抗元持续时间。", "确认约十年，自1265年坚持至1274年前后。"),
    (
        "causal",
        "对象为「公井」改称「贡井」的原因。",
        "因所产食盐上贡朝廷而改称贡井，得名逻辑成立。",
    ),
    (
        "causal",
        "对象为自贡盐业生产中心西移的原因。",
        "因富义盐井淡水渗入等而渐废，中心西移至自流井盐场，因果链完整。",
    ),
    (
        "default",
        "对象为「自贡」称谓出现的时间段。",
        "确认约在清代嘉庆、道光年间随自流井与贡井并立而出现。",
    ),
    (
        "default",
        "对象为清代「引岸专卖」制度的核心含义。",
        "确认官府配发盐引、限定销区与盐商经营范围、不得跨区域销售的核心含义。",
    ),
    (
        "default",
        "对象为清雍正七年自流井与贡井的行政分属。",
        "确认自流井归富顺、贡井归荣县，分设县丞署专管盐政。",
    ),
    (
        "default",
        "对象为自贡早期股票形式及其历史地位。",
        "确认「同盛井约」为中国历史上最早的股票之一。",
    ),
    (
        "default",
        "对象为自流井盐税与英法赔款的关联。",
        "确认部分赔款从自流井盐税中提取，并设有英法盐税协理官署。",
    ),
    (
        "causal",
        "对象为第一次川盐济楚的直接背景。",
        "太平天国致长江运输受阻、淮盐难运湘鄂，清廷改以川盐供应，因果链完整。",
    ),
    (
        "causal",
        "对象为第一次川盐济楚对自贡盐业的影响。",
        "市场扩大推动自流井、贡井盐业急剧扩张并进入鼎盛，因果链完整。",
    ),
    (
        "enum",
        "对象为清末自贡「四大家」盐商，需逐一提名避免遗漏。",
        "确认王三畏堂、李四友堂、胡慎怡堂、颜桂馨堂四家具齐。",
    ),
    (
        "default",
        "对象为外国学者对自流井盐业的评价。",
        "确认李希霍芬虽未亲赴但高度评价自流井井盐业规模。",
    ),
    (
        "default",
        "对象为1888年考察自贡并撰写考察记的外国人。",
        "确认美国传教士赫斐秋（弗吉尔·哈特）及《自流井考察记》的记载。",
    ),
    (
        "default",
        "对象为自贡会馆在历史上发挥的作用。",
        "确认会馆除祭祀乡神外还联络同乡、互助、调解纠纷，具移民同乡组织属性。",
    ),
    (
        "default",
        "对象为荣县在辛亥革命中的特殊地位。",
        "确认1911年荣县较早宣布脱离清廷独立。",
    ),
    (
        "enum",
        "对象为1911年荣县独立的主要领导者。",
        "确认吴玉章、王天杰等主要领导者具齐。",
    ),
    (
        "default",
        "对象为1911年自贡地方临时议事会的特殊性。",
        "确认其由自、贡两地同盟会员与地方人士筹建、商人力量强、尝试三权分立机构。",
    ),
    (
        "causal",
        "对象为军阀反复争夺自流井、贡井的原因。",
        "盐税收入巨大且为重要军费来源，控制盐场即可就近提税，因果链完整。",
    ),
    (
        "causal",
        "对象为善后借款合同对自贡盐业的影响。",
        "合同以盐税为担保，外国势力借盐务稽核机构介入盐税管理，因果链完整。",
    ),
    (
        "default",
        "对象为自贡设市最重要的经济基础。",
        "确认以自流井、贡井井盐产业及城市经济与人口集聚为设市基础。",
    ),
    (
        "causal",
        "对象为抗战时期第二次川盐济楚的背景。",
        "日本控制沿海盐区并切断海盐运输、内地盐荒，国民政府恢复川盐供应湘鄂，因果链完整。",
    ),
    (
        "default",
        "对象为孙明经1938年在自贡拍摄的纪录片。",
        "确认《自贡井盐》影片对井盐工业与城市景象的记录。",
    ),
    ("default", "对象为自贡正式设市的日期。", "确认1939年9月1日设市。"),
    (
        "default",
        "对象为自贡设市时的组成部分。",
        "确认以自流井、贡井两大盐业城域为主体组成。",
    ),
    (
        "causal",
        "对象为日军轰炸自贡的战略意图。",
        "通过「盐遮断」破坏中国内地食盐供应与战时经济，因果链完整。",
    ),
    (
        "default",
        "对象为1939年10月10日轰炸的称谓。",
        "确认「双十无差别轰炸」发生在自贡各界庆祝国庆日时。",
    ),
    ("default", "对象为日军两年间对自贡的投弹数量。", "确认约1,544枚。"),
    (
        "enum",
        "对象为自贡购买的三架飞机名称。",
        "确认「盐商号」「盐工号」「盐船号」三架具齐。",
    ),
    (
        "default",
        "对象为冯玉祥1943—1944年在自贡推动的活动。",
        "确认抗战献金运动及其募集资金支援前线的取向。",
    ),
    ("default", "对象为冯玉祥在自贡题写的四个大字。", "确认「还我河山」四字。"),
    (
        "default",
        "对象为战时第六保育院的位置。",
        "确认位于贡井天后宫，是当时四川规模很大的儿童保育院之一。",
    ),
    (
        "default",
        "对象为1943年与1944年自贡青年参加远征军的数量。",
        "确认1943年137人、1944年570人。",
    ),
    ("default", "对象为抗战后自贡的全国城市排名。", "确认约为全国第19大城市。"),
    (
        "default",
        "对象为《川康经济建设计划》对自贡的定位。",
        "确认其希望把自贡建为四川区域性工业中心并带动周边县市发展。",
    ),
    (
        "causal",
        "对象为1949年自贡政权更替的方式。",
        "市长甘绩丕等接应解放军并起义，使自贡得以较为迅速接管，因果链完整。",
    ),
    ("default", "对象为1950年川南行署驻地。", "确认最初驻自贡、后迁往泸州。"),
    ("default", "对象为1955年更名大安区的原区名。", "确认原为大坟堡区。"),
    (
        "default",
        "对象为1958年自贡提出的化工城市口号。",
        "确认「跨上千里马，奔向化工城」。",
    ),
    (
        "default",
        "对象为1958年前后自贡第一次产业结构调整的方向。",
        "确认由传统制盐向盐卤、天然气综合利用和化学工业延伸。",
    ),
    (
        "default",
        "对象为1959年建成的邓关盐厂的特殊地位。",
        "确认它为新中国成立后建设的第一座且唯一一座大型盐厂。",
    ),
    (
        "default",
        "对象为1960年自贡化工产品种类的变化。",
        "确认由1957年的8种增至1960年的60种。",
    ),
    ("default", "对象为1960年自贡化工产值占比。", "确认约占盐业含化工总产值的83.89%。"),
    (
        "causal",
        "对象为三线建设对自贡产业结构的改变。",
        "大批机械、化工、冶金、建材、科研单位内迁使自贡形成多门类工业体系，因果链完整。",
    ),
    (
        "default",
        "对象为内迁自贡的高校。",
        "确认华东化工学院（今华东理工大学）内迁自贡。",
    ),
    (
        "default",
        "对象为大山铺恐龙化石群最早发现重要化石的时间。",
        "确认1972年3月地质部第二普查大队在万年灯地区发现恐龙尾椎化石。",
    ),
    ("default", "对象为1979年划归自贡的县。", "确认荣县于1979年划归自贡。"),
    ("default", "对象为1983年划归自贡的县。", "确认富顺县于1983年划归自贡。"),
    ("default", "对象为1983年改名的区。", "确认郊区更名沿滩区。"),
    (
        "default",
        "对象为「四区两县」格局形成的时间。",
        "确认1983年富顺划归自贡、郊区更名沿滩区后基本形成四区两县。",
    ),
    (
        "default",
        "对象为1986年12月自贡获得的称号。",
        "确认为国务院公布的第二批「中国历史文化名城」。",
    ),
    ("default", "对象为首届自贡国际恐龙灯会经贸交易会的年份。", "确认为1987年。"),
    (
        "default",
        "对象为秦代五尺道与今自贡的关系。",
        "确认五尺道起点约在邓关镇附近，后发展为南方丝绸之路、茶马古道、盐马古道的重要商路。",
    ),
    (
        "default",
        "对象为自贡盐井开采前的基础工业。",
        "确认冶铁业及铁器冶炼技术为盐井开采提供了重要条件。",
    ),
    (
        "default",
        "对象为「金犍为、银富顺」中富顺所指的区域。",
        "确认主要指包括今自贡市区在内的富顺盐业经济区域。",
    ),
    (
        "enum",
        "对象为清初组织进入自贡的移民来源地。",
        "确认湖广、江西、广东、福建等地移民具齐。",
    ),
    (
        "default",
        "对象为清初「小票」的性质。",
        "确认为顺治后期开始对盐征收的税费，称「小票」。",
    ),
    (
        "default",
        "对象为盐商购买「盐引」的性质。",
        "确认为官府发给盐商的运销凭证，凭引在范围内合法运销。",
    ),
    (
        "default",
        "对象为建市前自流井、贡井的行政归属。",
        "确认自流井主要属富顺县、贡井主要属荣县。",
    ),
    (
        "causal",
        "对象为自贡被称为「双子城域」的原因。",
        "现代自贡由自流井与贡井两个独立发展的盐业城镇合并形成，因果链完整。",
    ),
]


# ----------------------------------------------------------
# build GOAP response
# ----------------------------------------------------------


def parse_response(response):
    concept = answer = None
    for line in response.split("\n"):
        if line.startswith("CONCEPT:"):
            concept = line.split(":", 1)[1].strip()
        elif line.startswith("ANSWER:"):
            answer = line.split(":", 1)[1].strip()
    return concept, answer


def build_actions(kind, concept, hint):
    actions = [ACT_DEFINE, ACT_RETRIEVE]
    if kind == "enum":
        actions.append(ACT_ENUMERATE)
    verify_action = ACT_CAUSE if kind == "causal" else ACT_VERIFY
    actions.append(
        dict(verify_action, eff=verify_action["eff"].replace("{hint}", hint))
    )
    actions.append(ACT_OUTPUT)
    return actions


def build_response(kind, concept, answer, state_hint, hint):
    actions = build_actions(kind, concept, hint)

    action_lines = []
    for act in actions:
        action_lines.append(f"- ACTION: {act['name']}")
        action_lines.append(f"  PRECONDITIONS: {act['pre']}")
        action_lines.append(f"  EFFECTS: {act['eff'].format(concept=concept)}")

    plan_lines = [
        f"{i}. {act['name']} → {act['eff'].format(concept=concept)}"
        for i, act in enumerate(actions, 1)
    ]

    return "\n".join(
        [
            f"GOAL: 就「{concept}」给出准确史实并完成回答。",
            "WORLD STATE:",
            f"- 用户就「{concept}」提出问题，需给出可核实的回答。",
            f"- {state_hint}",
            "ACTIONS:",
            *action_lines,
            "PLAN:",
            *plan_lines,
            f"CONCEPT: {concept}",
            f"ANSWER: {answer}",
        ]
    )


# ----------------------------------------------------------
# main
# ----------------------------------------------------------


def main():
    with open(SRC_JSONL, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    assert len(rows) == len(PROFILES), (
        f"profile count {len(PROFILES)} != rows {len(rows)}"
    )

    records = []
    for row, (kind, state_hint, hint) in zip(rows, PROFILES):
        concept, answer = parse_response(row["response"])
        assert concept and answer, row["instruction"]

        goap_response = build_response(kind, concept, answer, state_hint, hint)
        records.append({"instruction": row["instruction"], "response": goap_response})

    with open(DST_JSONL, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(rec, ensure_ascii=False) + "\n" for rec in records)

    print("rows:", len(records))
    for rec in records[:2]:
        print("=" * 60)
        print("Q:", rec["instruction"])
        print(rec["response"])


if __name__ == "__main__":
    main()
