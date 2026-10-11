
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

@dataclass(frozen=True)
class PseudoArticle:
    article_id: str
    title: str
    title_zh: str
    publisher: str
    published_at: dt.datetime
    content: str
    content_zh: str
    tags: list[str] = field(default_factory=list)

@dataclass(frozen=True)
class PseudoReport:
    report_id: str
    company: str
    ticker: str
    fiscal_period: str
    published_at: dt.datetime
    revenue_billion_cny: float
    net_income_billion_cny: float
    yoy_revenue_growth: float
    summary: str
    summary_zh: str

@dataclass(frozen=True)
class PseudoEntity:
    entity_id: str
    name: str
    name_zh: str
    entity_type: str
    attributes: dict[str, str] = field(default_factory=dict)

@dataclass(frozen=True)
class PseudoEdge:
    source_id: str
    target_id: str
    relation: str
    timestamp: dt.datetime
    confidence: float
    provenance: str

@dataclass(frozen=True)
class PseudoRule:
    rule_id: str
    body: list[str]
    head: str
    confidence: float
    support: float
    lift: float
    temporal_gap_days: int
    rationale: str
    rationale_zh: str

@dataclass(frozen=True)
class PseudoEvidenceStep:
    step_index: int
    entity: str
    relation: str
    target: str
    timestamp: dt.datetime
    source_id: str
    snippet: str

@dataclass(frozen=True)
class PseudoEvidenceChain:
    chain_id: str
    stock: str
    prediction_date: dt.date
    direction: str
    aggregated_confidence: float
    steps: list[PseudoEvidenceStep] = field(default_factory=list)
    supporting_rules: list[str] = field(default_factory=list)

def get_pseudo_articles() -> list[PseudoArticle]:
    return [
        PseudoArticle(
            article_id="art-001",
            title="BYD Surpasses Tesla in Global EV Sales for Q3 2024",
            title_zh="比亚迪2024年第三季度全球电动汽车销量超越特斯拉",
            publisher="Caixin Global",
            published_at=dt.datetime(2024, 10, 15, 8, 30),
            content=(
                "BYD Company Limited reported record-breaking electric vehicle "
                "deliveries of 1.13 million units in Q3 2024, surpassing Tesla's "
                "1.05 million units globally. Analysts attribute the surge to "
                "aggressive pricing strategies in Southeast Asian markets and "
                "strong demand for the Seal and Dolphin models in Europe."
            ),
            content_zh=(
                "比亚迪股份有限公司报告2024年第三季度电动汽车交付量创纪录，"
                "达到113万辆，超过特斯拉全球105万辆的交付量。分析师将这一增长"
                "归因于在东南亚市场的积极定价策略，以及海豹和海豚车型在欧洲的强劲需求。"
            ),
            tags=["BYD", "EV", "Tesla", "sales", "Q3-2024"],
        ),
        PseudoArticle(
            article_id="art-002",
            title="CATL Announces Solid-State Battery Breakthrough",
            title_zh="宁德时代宣布固态电池技术突破",
            publisher="Securities Times",
            published_at=dt.datetime(2024, 11, 3, 9, 0),
            content=(
                "Contemporary Amperex Technology (CATL) unveiled its next-generation "
                "condensed-matter battery achieving 500 Wh/kg energy density, "
                "slated for mass production in H2 2025. The breakthrough is expected "
                "to reshape supply chain dynamics for NIO, Li Auto, and BYD."
            ),
            content_zh=(
                "宁德时代发布新一代凝聚态电池，能量密度达到500 Wh/kg，"
                "计划于2025年下半年量产。这一突破预计将重塑蔚来、理想和比亚迪的供应链格局。"
            ),
            tags=["CATL", "battery", "solid-state", "NIO", "Li Auto", "BYD"],
        ),
        PseudoArticle(
            article_id="art-003",
            title="Alibaba Cloud Revenue Grows 12% Amid AI Demand",
            title_zh="阿里云收入在AI需求推动下增长12%",
            publisher="South China Morning Post",
            published_at=dt.datetime(2024, 8, 22, 10, 15),
            content=(
                "Alibaba Group's cloud computing division reported revenue of "
                "RMB 29.6 billion for Q2 FY2025, up 12% year-over-year, driven "
                "by enterprise AI workload migration. The Tongyi Qianwen large "
                "language model saw adoption across 90,000 enterprise customers."
            ),
            content_zh=(
                "阿里巴巴集团云计算部门报告2025财年第二季度收入为296亿元人民币，"
                "同比增长12%，主要受企业AI工作负载迁移推动。通义千问大模型已在"
                "9万家企业客户中得到应用。"
            ),
            tags=["Alibaba", "cloud", "AI", "revenue", "Tongyi"],
        ),
        PseudoArticle(
            article_id="art-004",
            title="Tencent Gaming Revenue Slows as Regulatory Scrutiny Increases",
            title_zh="腾讯游戏收入放缓 监管审查加强",
            publisher="Bloomberg China",
            published_at=dt.datetime(2024, 9, 10, 14, 0),
            content=(
                "Tencent Holdings' gaming segment grew only 3% in Q2 2024, "
                "the slowest pace in six quarters, as Chinese regulators tightened "
                "game approval processes. Honor of Kings daily active users "
                "declined 8% quarter-over-quarter amid competition from miHoYo."
            ),
            content_zh=(
                "腾讯控股游戏部门2024年第二季度仅增长3%，为六个季度以来最慢，"
                "原因是中国监管机构收紧了游戏审批流程。《王者荣耀》日活用户"
                "环比下降8%，面临来自米哈游的竞争。"
            ),
            tags=["Tencent", "gaming", "regulation", "Honor of Kings"],
        ),
        PseudoArticle(
            article_id="art-005",
            title="PetroChina Benefits from Rising Crude Oil Prices",
            title_zh="中国石油受益于原油价格上涨",
            publisher="China Securities Journal",
            published_at=dt.datetime(2024, 7, 5, 7, 45),
            content=(
                "PetroChina Company Limited reported a 15% increase in upstream "
                "operating profit for H1 2024, as Brent crude averaged $87/barrel. "
                "The company accelerated shale gas exploration in the Sichuan Basin, "
                "targeting 25 bcm annual production by 2026."
            ),
            content_zh=(
                "中国石油天然气股份有限公司报告2024年上半年上游营业利润增长15%，"
                "布伦特原油均价为每桶87美元。公司加快了四川盆地页岩气勘探，"
                "目标到2026年年产量达到250亿立方米。"
            ),
            tags=["PetroChina", "crude oil", "shale gas", "Sichuan"],
        ),
        PseudoArticle(
            article_id="art-006",
            title="BYD Supply Chain Disruption: Key Battery Supplier Factory Fire",
            title_zh="比亚迪供应链中断：核心电池供应商工厂火灾",
            publisher="Reuters China",
            published_at=dt.datetime(2024, 12, 1, 11, 30),
            content=(
                "A fire at Guoxuan High-Tech's Hefei plant disrupted lithium-iron-"
                "phosphate cathode supply to BYD, potentially affecting Q1 2025 "
                "production of 50,000 vehicles. BYD shares fell 4.2% on the news. "
                "Analysts expect CATL to capture displaced orders."
            ),
            content_zh=(
                "国轩高科合肥工厂发生火灾，磷酸铁锂正极材料供应中断，可能影响"
                "比亚迪2025年第一季度约5万辆汽车的生产。消息公布后比亚迪股价下跌4.2%。"
                "分析师预计宁德时代将承接转移订单。"
            ),
            tags=["BYD", "supply chain", "Guoxuan", "fire", "CATL"],
        ),
        PseudoArticle(
            article_id="art-007",
            title="Alibaba Announces $25B Buyback Program Extension",
            title_zh="阿里巴巴宣布扩大250亿美元回购计划",
            publisher="Caixin Global",
            published_at=dt.datetime(2025, 1, 15, 9, 0),
            content=(
                "Alibaba Group expanded its share repurchase program to $25 billion "
                "through March 2027, signaling management confidence. The company "
                "retired 3.2% of outstanding shares in 2024. Hong Kong-listed "
                "shares rose 6.1% on the announcement."
            ),
            content_zh=(
                "阿里巴巴集团将股票回购计划扩大至250亿美元，有效期至2027年3月，"
                "显示管理层信心。公司2024年注销了3.2%的流通股。消息公布后"
                "港股上涨6.1%。"
            ),
            tags=["Alibaba", "buyback", "shareholder return"],
        ),
        PseudoArticle(
            article_id="art-008",
            title="NIO Faces Liquidity Concerns as Cash Burn Accelerates",
            title_zh="蔚来现金消耗加速 面临流动性担忧",
            publisher="Financial Times China",
            published_at=dt.datetime(2025, 2, 10, 8, 0),
            content=(
                "NIO Inc.'s cash reserves fell to RMB 31.5 billion at end of Q4 "
                "2024, down from RMB 45.2 billion a year earlier. The company "
                "burned RMB 5.8 billion in Q4 alone, raising concerns about its "
                "ability to fund the ALPS and Firefly sub-brands without "
                "additional capital raises."
            ),
            content_zh=(
                "蔚来汽车2024年第四季度末现金储备降至315亿元人民币，较上年同期的"
                "452亿元大幅下降。公司仅第四季度就消耗了58亿元现金，引发市场对其"
                "能否在不额外融资的情况下支撑阿尔卑斯和萤火虫子品牌的担忧。"
            ),
            tags=["NIO", "liquidity", "cash burn", "ALPS", "Firefly"],
        ),
        PseudoArticle(
            article_id="art-009",
            title="CATL Signs 5-Year Supply Deal with BMW for European Plants",
            title_zh="宁德时代与宝马签署欧洲工厂五年供应协议",
            publisher="Yicai Global",
            published_at=dt.datetime(2025, 3, 1, 10, 0),
            content=(
                "CATL secured a five-year battery supply agreement with BMW AG "
                "worth an estimated EUR 12 billion, covering the German automaker's "
                "Neue Klasse platform. Production will come from CATL's Erfurt, "
                "Germany plant and a new facility in Debrecen, Hungary."
            ),
            content_zh=(
                "宁德时代与宝马集团签署了价值约120亿欧元的五年电池供应协议，"
                "覆盖宝马新世代平台。产能将来自宁德时代在德国埃尔福特的工厂"
                "以及匈牙利德布勒森的新工厂。"
            ),
            tags=["CATL", "BMW", "supply deal", "Europe", "Neue Klasse"],
        ),
        PseudoArticle(
            article_id="art-010",
            title="Tencent Invests $2B in Southeast Asian Fintech Expansion",
            title_zh="腾讯投资20亿美元拓展东南亚金融科技",
            publisher="DealStreetAsia",
            published_at=dt.datetime(2025, 3, 12, 13, 0),
            content=(
                "Tencent Holdings invested $2 billion across three Southeast Asian "
                "digital payment platforms, expanding WeChat Pay's footprint in "
                "Indonesia, Thailand, and Vietnam. The move intensifies competition "
                "with Ant Group's Alipay+ and Sea Limited's SeaMoney."
            ),
            content_zh=(
                "腾讯控股向三家东南亚数字支付平台投资20亿美元，扩大微信支付"
                "在印度尼西亚、泰国和越南的覆盖范围。此举加剧了与蚂蚁集团"
                "Alipay+和Sea集团SeaMoney的竞争。"
            ),
            tags=["Tencent", "fintech", "Southeast Asia", "WeChat Pay"],
        ),
    ]

def get_pseudo_reports() -> list[PseudoReport]:
    return [
        PseudoReport(
            report_id="rpt-001",
            company="BYD Company Limited",
            ticker="002594.SZ",
            fiscal_period="Q3 2024",
            published_at=dt.datetime(2024, 10, 29, 18, 0),
            revenue_billion_cny=201.1,
            net_income_billion_cny=11.6,
            yoy_revenue_growth=0.24,
            summary=(
                "BYD reported Q3 2024 revenue of RMB 201.1B (+24% YoY), with "
                "net income of RMB 11.6B. Automotive gross margin expanded to "
                "22.1% from 20.7% in Q3 2023, driven by vertical integration "
                "and declining lithium carbonate costs."
            ),
            summary_zh=(
                "比亚迪报告2024年第三季度收入2011亿元（同比+24%），净利润116亿元。"
                "汽车毛利率从2023年第三季度的20.7%扩大至22.1%，得益于垂直整合"
                "和碳酸锂成本下降。"
            ),
        ),
        PseudoReport(
            report_id="rpt-002",
            company="Alibaba Group Holding",
            ticker="9988.HK",
            fiscal_period="Q2 FY2025",
            published_at=dt.datetime(2024, 8, 15, 20, 0),
            revenue_billion_cny=243.2,
            net_income_billion_cny=24.3,
            yoy_revenue_growth=0.05,
            summary=(
                "Alibaba Q2 FY2025 revenue reached RMB 243.2B (+5% YoY). "
                "Cloud Intelligence Group revenue grew 12% to RMB 29.6B. "
                "Taobao/Tmall GMV stabilized after four quarters of decline, "
                "with 88VIP membership reaching 42 million users."
            ),
            summary_zh=(
                "阿里巴巴2025财年第二季度收入达到2432亿元（同比+5%）。"
                "云智能集团收入增长12%至296亿元。淘宝/天猫GMV在连续四个季度"
                "下降后企稳，88VIP会员达到4200万。"
            ),
        ),
        PseudoReport(
            report_id="rpt-003",
            company="CATL",
            ticker="300750.SZ",
            fiscal_period="FY 2024",
            published_at=dt.datetime(2025, 3, 5, 18, 30),
            revenue_billion_cny=410.5,
            net_income_billion_cny=50.7,
            yoy_revenue_growth=0.08,
            summary=(
                "CATL full-year 2024 revenue reached RMB 410.5B (+8% YoY), "
                "net income RMB 50.7B. Global market share in EV batteries "
                "held at 37.1%. R&D spending increased 22% to RMB 18.4B, "
                "focused on sodium-ion and solid-state technologies."
            ),
            summary_zh=(
                "宁德时代2024年全年收入达4105亿元（同比+8%），净利润507亿元。"
                "全球电动汽车电池市场份额保持在37.1%。研发投入增长22%至184亿元，"
                "重点投入钠离子和固态电池技术。"
            ),
        ),
        PseudoReport(
            report_id="rpt-004",
            company="Tencent Holdings",
            ticker="0700.HK",
            fiscal_period="Q3 2024",
            published_at=dt.datetime(2024, 11, 13, 19, 0),
            revenue_billion_cny=167.2,
            net_income_billion_cny=53.2,
            yoy_revenue_growth=0.09,
            summary=(
                "Tencent Q3 2024 revenue was RMB 167.2B (+9% YoY). Games "
                "revenue grew 3% to RMB 51.8B. Fintech and business services "
                "grew 12% to RMB 53.1B. WeChat MAU reached 1.38 billion. "
                "Video accounts ad revenue more than doubled YoY."
            ),
            summary_zh=(
                "腾讯2024年第三季度收入1672亿元（同比+9%）。游戏收入增长3%至518亿元。"
                "金融科技及企业服务增长12%至531亿元。微信月活用户达13.8亿。"
                "视频号广告收入同比翻倍以上。"
            ),
        ),
        PseudoReport(
            report_id="rpt-005",
            company="PetroChina Company Limited",
            ticker="601857.SH",
            fiscal_period="H1 2024",
            published_at=dt.datetime(2024, 8, 29, 18, 0),
            revenue_billion_cny=1_562.8,
            net_income_billion_cny=88.6,
            yoy_revenue_growth=0.03,
            summary=(
                "PetroChina H1 2024 revenue was RMB 1,562.8B (+3% YoY). "
                "Upstream segment operating profit rose 15% on higher crude "
                "realizations. Natural gas output grew 6.2% to 73.8 bcm. "
                "Refining margins compressed due to weak domestic fuel demand."
            ),
            summary_zh=(
                "中国石油2024年上半年收入15628亿元（同比+3%）。上游业务营业利润"
                "因原油价格上升增长15%。天然气产量增长6.2%至738亿立方米。"
                "炼油利润率因国内燃料需求疲软而收窄。"
            ),
        ),
    ]

def get_pseudo_entities() -> list[PseudoEntity]:
    return [
        PseudoEntity("ent-byd", "BYD", "比亚迪", "COMPANY",
                      {"ticker": "002594.SZ", "sector": "automotive"}),
        PseudoEntity("ent-catl", "CATL", "宁德时代", "COMPANY",
                      {"ticker": "300750.SZ", "sector": "battery"}),
        PseudoEntity("ent-alibaba", "Alibaba", "阿里巴巴", "COMPANY",
                      {"ticker": "9988.HK", "sector": "technology"}),
        PseudoEntity("ent-tencent", "Tencent", "腾讯", "COMPANY",
                      {"ticker": "0700.HK", "sector": "technology"}),
        PseudoEntity("ent-petrochina", "PetroChina", "中国石油", "COMPANY",
                      {"ticker": "601857.SH", "sector": "energy"}),
        PseudoEntity("ent-nio", "NIO", "蔚来", "COMPANY",
                      {"ticker": "NIO", "sector": "automotive"}),
        PseudoEntity("ent-guoxuan", "Guoxuan High-Tech", "国轩高科", "COMPANY",
                      {"ticker": "002074.SZ", "sector": "battery"}),
        PseudoEntity("ent-bmw", "BMW", "宝马", "COMPANY",
                      {"ticker": "BMW.DE", "sector": "automotive"}),
        PseudoEntity("ent-ev-sales-record", "Q3 2024 EV Sales Record",
                      "2024年三季度电动车销售纪录", "EVENT", {}),
        PseudoEntity("ent-solid-state", "Solid-State Battery Breakthrough",
                      "固态电池技术突破", "EVENT", {}),
        PseudoEntity("ent-factory-fire", "Guoxuan Hefei Factory Fire",
                      "国轩合肥工厂火灾", "EVENT", {}),
        PseudoEntity("ent-brent-crude", "Brent Crude Oil", "布伦特原油", "METRIC",
                      {"unit": "USD/barrel"}),
        PseudoEntity("ent-licarbonate", "Lithium Carbonate Price",
                      "碳酸锂价格", "METRIC", {"unit": "CNY/tonne"}),
    ]

def get_pseudo_edges() -> list[PseudoEdge]:
    return [
        PseudoEdge("ent-byd", "ent-ev-sales-record", "achieved",
                    dt.datetime(2024, 10, 15), 0.95, "art-001"),
        PseudoEdge("ent-catl", "ent-solid-state", "announced",
                    dt.datetime(2024, 11, 3), 0.92, "art-002"),
        PseudoEdge("ent-catl", "ent-byd", "supplies_battery_to",
                    dt.datetime(2024, 11, 3), 0.88, "art-002"),
        PseudoEdge("ent-catl", "ent-nio", "supplies_battery_to",
                    dt.datetime(2024, 11, 3), 0.85, "art-002"),
        PseudoEdge("ent-guoxuan", "ent-factory-fire", "experienced",
                    dt.datetime(2024, 12, 1), 0.97, "art-006"),
        PseudoEdge("ent-factory-fire", "ent-byd", "disrupted_supply_to",
                    dt.datetime(2024, 12, 1), 0.93, "art-006"),
        PseudoEdge("ent-catl", "ent-byd", "captures_displaced_orders_from",
                    dt.datetime(2024, 12, 5), 0.70, "art-006"),
        PseudoEdge("ent-alibaba", "ent-alibaba", "revenue_growth",
                    dt.datetime(2024, 8, 22), 0.90, "art-003"),
        PseudoEdge("ent-alibaba", "ent-alibaba", "share_buyback",
                    dt.datetime(2025, 1, 15), 0.95, "art-007"),
        PseudoEdge("ent-tencent", "ent-tencent", "gaming_revenue_slowdown",
                    dt.datetime(2024, 9, 10), 0.88, "art-004"),
        PseudoEdge("ent-tencent", "ent-tencent", "fintech_expansion",
                    dt.datetime(2025, 3, 12), 0.85, "art-010"),
        PseudoEdge("ent-petrochina", "ent-brent-crude", "benefits_from",
                    dt.datetime(2024, 7, 5), 0.91, "art-005"),
        PseudoEdge("ent-catl", "ent-bmw", "signed_supply_deal_with",
                    dt.datetime(2025, 3, 1), 0.96, "art-009"),
        PseudoEdge("ent-nio", "ent-nio", "cash_burn_acceleration",
                    dt.datetime(2025, 2, 10), 0.90, "art-008"),
        PseudoEdge("ent-byd", "ent-byd", "margin_expansion",
                    dt.datetime(2024, 10, 29), 0.93, "rpt-001"),
        PseudoEdge("ent-catl", "ent-catl", "rd_spending_increase",
                    dt.datetime(2025, 3, 5), 0.91, "rpt-003"),
    ]

def get_pseudo_rules() -> list[PseudoRule]:
    return [
        PseudoRule(
            rule_id="rule-001",
            body=["supply_chain_disruption(A, B)", "revenue_decline(B)"],
            head="stock_down(B, t+5d)",
            confidence=0.72,
            support=0.15,
            lift=2.1,
            temporal_gap_days=5,
            rationale=(
                "Supply chain disruptions propagate downstream: when a key "
                "supplier experiences disruption and the affected company shows "
                "revenue decline, stock price drops within 5 trading days."
            ),
            rationale_zh=(
                "供应链中断向下游传导：当关键供应商出现中断且受影响公司出现"
                "收入下降时，股价在5个交易日内下跌。"
            ),
        ),
        PseudoRule(
            rule_id="rule-002",
            body=["share_buyback(A)", "revenue_growth(A)"],
            head="stock_up(A, t+10d)",
            confidence=0.78,
            support=0.22,
            lift=1.9,
            temporal_gap_days=10,
            rationale=(
                "Share buybacks combined with revenue growth signal management "
                "confidence and fundamental strength, leading to stock appreciation."
            ),
            rationale_zh=(
                "股票回购叠加收入增长，表明管理层信心和基本面强劲，"
                "推动股价上涨。"
            ),
        ),
        PseudoRule(
            rule_id="rule-003",
            body=["technology_breakthrough(A)", "market_leader(A)"],
            head="stock_up(A, t+15d)",
            confidence=0.65,
            support=0.12,
            lift=1.7,
            temporal_gap_days=15,
            rationale=(
                "When a market-leading company announces a technology breakthrough, "
                "institutional investors increase positions, driving stock up."
            ),
            rationale_zh=(
                "当市场领先企业宣布技术突破时，机构投资者增加持仓，推动股价上涨。"
            ),
        ),
        PseudoRule(
            rule_id="rule-004",
            body=["cash_burn_acceleration(A)", "no_new_funding(A)"],
            head="stock_down(A, t+7d)",
            confidence=0.81,
            support=0.18,
            lift=2.4,
            temporal_gap_days=7,
            rationale=(
                "Accelerating cash burn without new funding raises dilution or "
                "insolvency fears, triggering selling pressure within a week."
            ),
            rationale_zh=(
                "现金消耗加速而无新融资，引发稀释或偿付能力担忧，"
                "一周内触发卖压。"
            ),
        ),
        PseudoRule(
            rule_id="rule-005",
            body=["regulatory_tightening(sector)", "revenue_slowdown(A, sector)"],
            head="stock_down(A, t+20d)",
            confidence=0.68,
            support=0.20,
            lift=1.8,
            temporal_gap_days=20,
            rationale=(
                "Sector-level regulatory tightening combined with company-specific "
                "revenue slowdown leads to sustained stock decline over 20 days."
            ),
            rationale_zh=(
                "行业监管收紧叠加公司收入放缓，导致股价在20天内持续下跌。"
            ),
        ),
        PseudoRule(
            rule_id="rule-006",
            body=["commodity_price_rise(X)", "upstream_exposure(A, X)"],
            head="stock_up(A, t+3d)",
            confidence=0.74,
            support=0.25,
            lift=2.0,
            temporal_gap_days=3,
            rationale=(
                "Rising commodity prices directly boost upstream producers' "
                "earnings expectations, leading to rapid stock re-rating."
            ),
            rationale_zh=(
                "大宗商品价格上涨直接提升上游生产商的盈利预期，"
                "导致股价快速重估。"
            ),
        ),
        PseudoRule(
            rule_id="rule-007",
            body=["major_contract_signed(A, B)", "capacity_expansion(A)"],
            head="stock_up(A, t+5d)",
            confidence=0.76,
            support=0.14,
            lift=2.2,
            temporal_gap_days=5,
            rationale=(
                "Signing a major supply contract while expanding capacity signals "
                "locked-in future revenue, boosting investor confidence."
            ),
            rationale_zh=(
                "签署重大供应合同同时扩大产能，意味着未来收入锁定，"
                "提振投资者信心。"
            ),
        ),
        PseudoRule(
            rule_id="rule-008",
            body=["competitor_disruption(B)", "same_sector(A, B)", "capacity_available(A)"],
            head="stock_up(A, t+10d)",
            confidence=0.63,
            support=0.11,
            lift=1.6,
            temporal_gap_days=10,
            rationale=(
                "When a competitor faces disruption and the company has available "
                "capacity in the same sector, it captures displaced demand."
            ),
            rationale_zh=(
                "当竞争对手遭遇中断且同行业公司有可用产能时，"
                "该公司将承接转移需求。"
            ),
        ),
    ]

def get_pseudo_evidence_chains() -> list[PseudoEvidenceChain]:
    return [
        PseudoEvidenceChain(
            chain_id="chain-001",
            stock="BYD (002594.SZ)",
            prediction_date=dt.date(2024, 12, 6),
            direction="DOWN",
            aggregated_confidence=0.71,
            steps=[
                PseudoEvidenceStep(
                    0, "Guoxuan High-Tech", "experienced",
                    "Factory Fire (Hefei Plant)",
                    dt.datetime(2024, 12, 1), "art-006",
                    "A fire at Guoxuan High-Tech's Hefei plant disrupted "
                    "lithium-iron-phosphate cathode supply...",
                ),
                PseudoEvidenceStep(
                    1, "Factory Fire", "disrupted_supply_to", "BYD",
                    dt.datetime(2024, 12, 1), "art-006",
                    "...potentially affecting Q1 2025 production of 50,000 vehicles.",
                ),
                PseudoEvidenceStep(
                    2, "BYD", "stock_price_reaction", "DOWN -4.2%",
                    dt.datetime(2024, 12, 2), "art-006",
                    "BYD shares fell 4.2% on the news.",
                ),
            ],
            supporting_rules=["rule-001"],
        ),
        PseudoEvidenceChain(
            chain_id="chain-002",
            stock="CATL (300750.SZ)",
            prediction_date=dt.date(2025, 3, 10),
            direction="UP",
            aggregated_confidence=0.77,
            steps=[
                PseudoEvidenceStep(
                    0, "CATL", "announced", "Solid-State Battery Breakthrough",
                    dt.datetime(2024, 11, 3), "art-002",
                    "CATL unveiled next-generation condensed-matter battery "
                    "achieving 500 Wh/kg energy density...",
                ),
                PseudoEvidenceStep(
                    1, "CATL", "signed_supply_deal_with", "BMW",
                    dt.datetime(2025, 3, 1), "art-009",
                    "CATL secured a five-year battery supply agreement with "
                    "BMW AG worth an estimated EUR 12 billion...",
                ),
                PseudoEvidenceStep(
                    2, "CATL", "rd_spending_increase", "RMB 18.4B (+22% YoY)",
                    dt.datetime(2025, 3, 5), "rpt-003",
                    "R&D spending increased 22% to RMB 18.4B, focused on "
                    "sodium-ion and solid-state technologies.",
                ),
            ],
            supporting_rules=["rule-003", "rule-007"],
        ),
        PseudoEvidenceChain(
            chain_id="chain-003",
            stock="NIO (NIO)",
            prediction_date=dt.date(2025, 2, 17),
            direction="DOWN",
            aggregated_confidence=0.74,
            steps=[
                PseudoEvidenceStep(
                    0, "NIO", "cash_burn_acceleration", "RMB 5.8B in Q4 2024",
                    dt.datetime(2025, 2, 10), "art-008",
                    "NIO Inc.'s cash reserves fell to RMB 31.5 billion at "
                    "end of Q4 2024, down from RMB 45.2 billion...",
                ),
                PseudoEvidenceStep(
                    1, "NIO", "sub_brand_cost_pressure",
                    "ALPS and Firefly launch costs",
                    dt.datetime(2025, 2, 10), "art-008",
                    "...raising concerns about its ability to fund the ALPS "
                    "and Firefly sub-brands without additional capital raises.",
                ),
            ],
            supporting_rules=["rule-004"],
        ),
    ]
