"""
################################################################################
#                                                                              #
#     ABM讲义实验平台 - 统一交互式仿真界面                                       #
#     ABM Lecture Experiment Platform - Unified Interactive Interface          #
#                                                                              #
################################################################################

================================================================================
一、整体设计思路 / Overall Design Philosophy
================================================================================

本程序整合了《基于智能体建模(ABM)》讲义中的全部实验模型，提供统一的
交互式仿真界面，支持参数配置、结果可视化和数据导出。

【集成模型清单 / Integrated Models】

┌─────────────────────────────────────────────────────────────────────────────┐
│  第6章: 农民作物选择模型 (6_2.py)                                            │
│         - 风险偏好异质性 + 社会学习机制                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  第7章: 农业生产决策模型                                                      │
│         - 7.1 农民决策与环境影响模型 (环境意识+碳价格)                         │
│         - 7.2 灌溉与水资源管理模型 (水分胁迫+灌溉策略)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  第8章: 农业金融模型                                                          │
│         - 8.1 农业保险采纳模型 (期望效用+政府补贴)                             │
│         - 8.2 农村信贷风险评估模型 (信用评级+违约风险)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  第10章: 技术与水文模型                                                       │
│         - 10.2 农业技术扩散模型 (Rogers创新扩散+S型曲线)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  第12章: 银行体系风险传染模型                                                  │
│         - 基于张亮(2017)博士论文理论框架                                       │
│         - 网络拓扑(无标度/小世界/随机) + 冲击传导                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  专题: 洪水风险与保险ABM模型 (Flood Re机制)                                    │
└─────────────────────────────────────────────────────────────────────────────┘

【运行方式 / How to Run】

    cd 讲义
    streamlit run abm_experiment_platform.py

【依赖安装 / Install Dependencies】

    pip install streamlit pandas numpy matplotlib mesa networkx

【程序结构 / Program Structure】

    第1部分: 模型导入与配置 (Lines 80-150)
    第2部分: 侧边栏 - 模型选择与参数配置 (Lines 150-500)
    第3部分: 各模型运行函数 (Lines 500-900)
    第4部分: 可视化与结果展示 (Lines 900-1200)
    第5部分: 数据导出功能 (Lines 1200-1300)

作者: 肖诗顺
版本: v2.0 (独立运行版 - 全部模型代码已内置)
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
import networkx as nx
import sys
import os
from mesa import Model, Agent
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

# 添加粮食市场模型目录到路径
_grain_model_path = os.path.join(os.path.dirname(__file__), "..", "粮食市场模型")
if os.path.exists(_grain_model_path):
    sys.path.insert(0, os.path.abspath(_grain_model_path))

# 尝试导入完整粮食市场模型和校准参数
try:
    from grain_market_mvp_model import (
        GrainMarketModel as FullGrainMarketModel,
        FarmerAgent, GovernmentAgent, InsuranceFirm, RuralBank, PolicyBank,
        ForeignSectorAgent, DomesticBuyer, CropType, ProducerType, 
        CreditGrade, InsuranceProductType, BuyerType
    )
    from calibration_params import CALIBRATED_PARAMS, CalibratedParameters
    GRAIN_MODEL_AVAILABLE = True
except ImportError as e:
    GRAIN_MODEL_AVAILABLE = False
    print(f"粮食市场完整模型未加载: {e}")

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 页面配置
# =============================================================================
st.set_page_config(
    page_title="ABM讲义实验平台",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 字体大小设置（通过CSS注入）
# =============================================================================
# 字体大小映射表（中文字号 -> pt值）
FONT_SIZE_MAP = {
    "小五号 (9pt)": 9,
    "五号 (10.5pt)": 10.5,
    "小四号 (12pt)": 12,
    "四号 (14pt)": 14,  # 默认
    "小三号 (15pt)": 15,
    "三号 (16pt)": 16,
    "小二号 (18pt)": 18,
    "二号 (22pt)": 22,
}

# 从session_state获取字体大小，默认四号
if 'font_size' not in st.session_state:
    st.session_state.font_size = "四号 (14pt)"

# 从session_state获取语言设置，默认中文
if 'language' not in st.session_state:
    st.session_state.language = "中文"

# 获取当前字体大小
current_font_pt = FONT_SIZE_MAP[st.session_state.font_size]

# =============================================================================
# 双语字典 / Bilingual Dictionary
# =============================================================================
TRANSLATIONS = {
    "中文": {
        "page_title": "ABM讲义实验平台",
        "page_subtitle": "Agent-Based Modeling Lecture Experiment Platform",
        "experiment_config": "🎛️ 实验配置",
        "display_settings": "🔤 显示设置",
        "font_size_label": "字体大小",
        "select_model": "1️⃣ 选择实验模型",
        "common_params": "2️⃣ 通用参数",
        "simulation_steps": "仿真步数",
        "random_seed": "随机种子",
        "model_params": "3️⃣ 模型专属参数",
        "run_simulation": "🚀 运行仿真",
        "simulation_results": "📈 仿真结果",
        "theory_explanation": "📖 理论说明",
        "detailed_data": "📋 详细数据",
        "data_export": "💾 数据导出",
        "export_csv": "导出为CSV",
        "export_excel": "导出为Excel",
        "export_json": "导出为JSON",
        "initial_prompt": "👈 请在左侧边栏配置参数，然后点击 **运行仿真** 按钮开始实验",
        "language": "语言 / Language",
        "chinese": "中文",
        "english": "English",
        "keywords": "关键词",
        "chapter": "章节",
        "basic_params": "基本参数",
        "extended_params": "扩展实验参数",
        "robustness_test": "稳健性检验",
        "enable_monte_carlo": "启用蒙特卡洛实验",
        "monte_carlo_runs": "重复运行次数",
        "show_network_viz": "显示传染网络可视化",
        "enable_bailout": "启用央行流动性救助",
        "bailout_intensity": "救助强度 (% of 资本)",
        "network_density": "BA网络密度参数 m",
        "lgd": "违约损失率 LGD (%)",
        "car_threshold": "资本充足率阈值 (%)",
        "stress_increment": "压力累积速率",
        "monte_carlo_results": "蒙特卡洛实验结果",
        "network_viz": "传染网络可视化",
        "core_metrics": "核心指标",
        "bailout_stats": "央行救助统计",
        "key_indicators": "关键指标",
        "grain_market_warning": "⎡ 完整粮食市场模型未加载，使用简化版本",
        "grain_market_base_params": "📊 基础参数 (校准数据来源: 全国人大农业法执法检查报告2024)",
        "grain_market_policy_switches": "🔧 政策模块开关",
        "grain_market_tariff": "📦 关税政策 (数据来源: 财政部2024年关税调整方案)",
        "grain_market_insurance": "🛡️ 保险政策 (校准: 财政部财金〔2023〕59号)",
        "grain_market_credit": "🏦 信贷政策 (校准: 人行普惠金融报告2024)",
        "grain_market_policy_bank": "🏛️ 农发行收购贷款 (校准: 农业发展银行官网)",
        "grain_market_subsidy": "💰 生产补贴 (元/公顷)",
        "grain_market_scenario": "🎯 政策情景",
        "grain_market_farmer_ratio": "👨‍🌾 农户自缴比例",
        "grain_market_coverage": "三大粮食作物保险覆盖率校准值: 82%",
        "n_farmers": "农户数量",
        "grid_size": "网格大小",
        "social_learning_prob": "社会学习概率",
        "carbon_price": "碳价格 (元/吨CO2)",
        "fertilizer_cost": "化肥成本 (元/公顷)",
        "water_scenario": "水资源情景",
        "irrigation_cost_factor": "灌溉成本系数",
        "premium_rate": "保费率 (%)",
        "subsidy_rate": "政府补贴率 (%)",
        "disaster_prob": "灾害发生概率 (%)",
        "base_interest_rate": "基准利率 (%)",
        "credit_threshold": "授信阈值 (信用分)",
        "initial_adopters": "初始采用者比例 (%)",
        "n_banks": "银行数量",
        "network_topology": "网络拓扑结构",
        "shock_scenario": "冲击情景",
        "view_all_models": "查看所有模型",
        "model_name": "模型名称",
        "code_file": "代码文件",
        "footer": "ABM讲义实验平台 | 作者: 肖诗顺 | 版本: v2.0",
        "adequate_water": "充足水资源",
        "moderate_water_shortage": "中等缺水",
        "severe_water_shortage": "严重缺水",
        "scale_free_network": "无标度网络 (scale_free)",
        "small_world_network": "小世界网络 (small_world)",
        "random_network": "随机网络 (random)",
        "single_bank_default": "单个银行违约",
        "multiple_banks_default": "多银行同时违约",
        "distributed_shocks": "多点分散冲击",
        "macroeconomic_shock": "宏观经济冲击",
        "progressive_shock": "渐进式冲击",
        "sufficient_rainfall": "充足降雨",
        "moderate_drought": "中度干旱",
        "severe_drought": "严重干旱",
        "view_full_data": "查看完整数据",
        "step": "步数",
        "crop_traditional": "传统作物",
        "crop_new": "新作物",
        "crop_diversified": "多样化",
        "mean_wealth": "平均财富",
        "std_wealth": "财富标准差",
        "traditional_ratio": "传统作物比例",
        "new_crop_ratio": "新作物比例",
        "diversified_ratio": "多样化比例",
        "eco_farmers_ratio": "环保农户比例",
        "total_emissions": "总排放量",
        "insurance_adoption": "保险采纳率",
        "mean_wealth_insured": "平均财富(参保)",
        "mean_wealth_uninsured": "平均财富(未参保)",
        "credit_approval_rate": "信贷审批率",
        "default_rate": "违约率",
        "mean_credit_score": "平均信用分",
        "adoption_rate": "采纳率",
        "cumulative_adopters": "累计采用者",
        "failed_banks": "失败银行数",
        "contagion_rounds": "传染轮数",
        "total_loss": "总损失",
        "max_single_loss": "最大单笔损失",
        "flood_depth": "洪水深度 (m)",
        "insured_loss": "保险赔付损失",
        "uninsured_loss": "未保险损失",
        "reinsurance_payout": "再保险赔付",
        "gdp": "GDP",
        "unemployment": "失业率",
        "inflation": "通胀率",
        "government_debt": "政府债务",
        "grain_price": "粮食价格",
        "total_production": "总产量",
        "farmer_income": "农户收入",
        "insurance_premium": "保险保费",
        "insurance_payout": "保险赔付",
        "credit_amount": "信贷金额",
        "default_amount": "违约金额",
        "subsidy_amount": "补贴金额",
        "tariff_revenue": "关税收入",
        "policy_bank_loan": "政策性银行贷款"
    },
    "English": {
        "page_title": "ABM Lecture Experiment Platform",
        "page_subtitle": "Agent-Based Modeling Lecture Experiment Platform",
        "experiment_config": "🎛️ Experiment Configuration",
        "display_settings": "🔤 Display Settings",
        "font_size_label": "Font Size",
        "select_model": "1️⃣ Select Experiment Model",
        "common_params": "2️⃣ Common Parameters",
        "simulation_steps": "Simulation Steps",
        "random_seed": "Random Seed",
        "model_params": "3️⃣ Model-Specific Parameters",
        "run_simulation": "🚀 Run Simulation",
        "simulation_results": "📈 Simulation Results",
        "theory_explanation": "📖 Theory Explanation",
        "detailed_data": "📋 Detailed Data",
        "data_export": "💾 Data Export",
        "export_csv": "Export as CSV",
        "export_excel": "Export as Excel",
        "export_json": "Export as JSON",
        "initial_prompt": "👈 Configure parameters in the left sidebar, then click **Run Simulation** to start the experiment",
        "language": "语言 / Language",
        "chinese": "中文",
        "english": "English",
        "keywords": "Keywords",
        "chapter": "Chapter",
        "basic_params": "Basic Parameters",
        "extended_params": "Extended Experiment Parameters",
        "robustness_test": "Robustness Test",
        "enable_monte_carlo": "Enable Monte Carlo Experiment",
        "monte_carlo_runs": "Number of Repetitions",
        "show_network_viz": "Show Contagion Network Visualization",
        "enable_bailout": "Enable Central Bank Liquidity Bailout",
        "bailout_intensity": "Bailout Intensity (% of Capital)",
        "network_density": "BA Network Density Parameter m",
        "lgd": "Loss Given Default LGD (%)",
        "car_threshold": "Capital Adequacy Ratio Threshold (%)",
        "stress_increment": "Stress Accumulation Rate",
        "monte_carlo_results": "Monte Carlo Experiment Results",
        "network_viz": "Contagion Network Visualization",
        "core_metrics": "Core Metrics",
        "bailout_stats": "Central Bank Bailout Statistics",
        "key_indicators": "Key Indicators",
        "grain_market_warning": "⎡ Full grain market model not loaded, using simplified version",
        "grain_market_base_params": "📊 Basic Parameters (Calibration Data Source: NPC Agriculture Law Enforcement Inspection Report 2024)",
        "grain_market_policy_switches": "🔧 Policy Module Switches",
        "grain_market_tariff": "📦 Tariff Policy (Data Source: MOF 2024 Tariff Adjustment Plan)",
        "grain_market_insurance": "🛡️ Insurance Policy (Calibration: MOF Finance [2023] No. 59)",
        "grain_market_credit": "🏦 Credit Policy (Calibration: PBOC Inclusive Finance Report 2024)",
        "grain_market_policy_bank": "🏛️ ADBC Acquisition Loan (Calibration: Agricultural Development Bank of China Official Website)",
        "grain_market_subsidy": "💰 Production Subsidy (CNY/hectare)",
        "grain_market_scenario": "🎯 Policy Scenario",
        "grain_market_farmer_ratio": "👨‍🌾 Farmer Self-Payment Ratio",
        "grain_market_coverage": "Three Major Grain Crops Insurance Coverage Calibration: 82%",
        "n_farmers": "Number of Farmers",
        "grid_size": "Grid Size",
        "social_learning_prob": "Social Learning Probability",
        "carbon_price": "Carbon Price (CNY/ton CO2)",
        "fertilizer_cost": "Fertilizer Cost (CNY/hectare)",
        "water_scenario": "Water Scenario",
        "irrigation_cost_factor": "Irrigation Cost Factor",
        "premium_rate": "Premium Rate (%)",
        "subsidy_rate": "Government Subsidy Rate (%)",
        "disaster_prob": "Disaster Probability (%)",
        "base_interest_rate": "Base Interest Rate (%)",
        "credit_threshold": "Credit Threshold (Credit Score)",
        "initial_adopters": "Initial Adopter Ratio (%)",
        "n_banks": "Number of Banks",
        "network_topology": "Network Topology",
        "shock_scenario": "Shock Scenario",
        "view_all_models": "View All Models",
        "model_name": "Model Name",
        "code_file": "Code File",
        "footer": "ABM Lecture Experiment Platform | Author: Xiao Shishun | Version: v2.0",
        "adequate_water": "Adequate Water Resources",
        "moderate_water_shortage": "Moderate Water Shortage",
        "severe_water_shortage": "Severe Water Shortage",
        "scale_free_network": "Scale-Free Network (scale_free)",
        "small_world_network": "Small-World Network (small_world)",
        "random_network": "Random Network (random)",
        "single_bank_default": "Single Bank Default",
        "multiple_banks_default": "Multiple Banks Default",
        "distributed_shocks": "Distributed Shocks",
        "macroeconomic_shock": "Macroeconomic Shock",
        "progressive_shock": "Progressive Shock",
        "sufficient_rainfall": "Sufficient Rainfall",
        "moderate_drought": "Moderate Drought",
        "severe_drought": "Severe Drought",
        "view_full_data": "View Full Data",
        "step": "Step",
        "crop_traditional": "Traditional Crop",
        "crop_new": "New Crop",
        "crop_diversified": "Diversified",
        "mean_wealth": "Mean Wealth",
        "std_wealth": "Std Wealth",
        "traditional_ratio": "Traditional Ratio",
        "new_crop_ratio": "New Crop Ratio",
        "diversified_ratio": "Diversified Ratio",
        "eco_farmers_ratio": "Eco-Farmers Ratio",
        "total_emissions": "Total Emissions",
        "insurance_adoption": "Insurance Adoption",
        "mean_wealth_insured": "Mean Wealth (Insured)",
        "mean_wealth_uninsured": "Mean Wealth (Uninsured)",
        "credit_approval_rate": "Credit Approval Rate",
        "default_rate": "Default Rate",
        "mean_credit_score": "Mean Credit Score",
        "adoption_rate": "Adoption Rate",
        "cumulative_adopters": "Cumulative Adopters",
        "failed_banks": "Failed Banks",
        "contagion_rounds": "Contagion Rounds",
        "total_loss": "Total Loss",
        "max_single_loss": "Max Single Loss",
        "flood_depth": "Flood Depth (m)",
        "insured_loss": "Insured Loss",
        "uninsured_loss": "Uninsured Loss",
        "reinsurance_payout": "Reinsurance Payout",
        "gdp": "GDP",
        "unemployment": "Unemployment Rate",
        "inflation": "Inflation Rate",
        "government_debt": "Government Debt",
        "grain_price": "Grain Price",
        "total_production": "Total Production",
        "farmer_income": "Farmer Income",
        "insurance_premium": "Insurance Premium",
        "insurance_payout": "Insurance Payout",
        "credit_amount": "Credit Amount",
        "default_amount": "Default Amount",
        "subsidy_amount": "Subsidy Amount",
        "tariff_revenue": "Tariff Revenue",
        "policy_bank_loan": "Policy Bank Loan"
    }
}

# 辅助函数：获取当前语言的翻译
def t(key):
    """Get translation for the given key in current language"""
    lang = st.session_state.language
    return TRANSLATIONS[lang].get(key, key)

# 注入自定义CSS样式
st.markdown(f"""
<style>
    /* 主内容区字体大小 */
    .stMarkdown, .stText, .element-container p, 
    .stDataFrame, .stTable, div[data-testid="stExpander"] {{
        font-size: {current_font_pt}pt !important;
    }}
    /* 标题保持相对比例 */
    h1 {{ font-size: {current_font_pt * 1.8}pt !important; }}
    h2 {{ font-size: {current_font_pt * 1.5}pt !important; }}
    h3 {{ font-size: {current_font_pt * 1.3}pt !important; }}
    h4 {{ font-size: {current_font_pt * 1.1}pt !important; }}
    /* 代码块 */
    code, pre {{ font-size: {current_font_pt * 0.9}pt !important; }}
    /* 侧边栏 */
    .css-1d391kg, [data-testid="stSidebar"] {{
        font-size: {current_font_pt * 0.9}pt !important;
    }}
    /* 表格内容 */
    .dataframe td, .dataframe th {{
        font-size: {current_font_pt}pt !important;
    }}
</style>
""", unsafe_allow_html=True)

# 标题
st.title(f"🔬 {t('page_title')}")
st.markdown(f"**{t('page_subtitle')}**")
st.markdown("---")

# =============================================================================
# 双语模型目录 / Bilingual Model Catalog
# =============================================================================
# =============================================================================
# 模型名称双语映射
# =============================================================================
MODEL_NAMES = {
    "中文": {
        "6.2 农民作物选择模型": "6.2 农民作物选择模型",
        "7.1 农民决策与环境影响模型": "7.1 农民决策与环境影响模型",
        "7.2 灌溉与水资源管理模型": "7.2 灌溉与水资源管理模型",
        "8.1 农业保险采纳模型": "8.1 农业保险采纳模型",
        "8.2 农村信贷风险评估模型": "8.2 农村信贷风险评估模型",
        "10.2 农业技术扩散模型": "10.2 农业技术扩散模型",
        "12.2-12.5 银行风险传染模型": "12.2-12.5 银行风险传染模型",
        "专题: 洪水风险与保险ABM模型": "专题: 洪水风险与保险ABM模型",
        "专题: 欧元区经济危机AB-SFC模型": "专题: 欧元区经济危机AB-SFC模型",
        "专题: 粮食市场政策仿真模型": "专题: 粮食市场政策仿真模型"
    },
    "English": {
        "6.2 农民作物选择模型": "6.2 Farmer Crop Selection Model",
        "7.1 农民决策与环境影响模型": "7.1 Farmer Decision & Environmental Impact Model",
        "7.2 灌溉与水资源管理模型": "7.2 Irrigation & Water Resource Management Model",
        "8.1 农业保险采纳模型": "8.1 Agricultural Insurance Adoption Model",
        "8.2 农村信贷风险评估模型": "8.2 Rural Credit Risk Assessment Model",
        "10.2 农业技术扩散模型": "10.2 Agricultural Technology Diffusion Model",
        "12.2-12.5 银行风险传染模型": "12.2-12.5 Banking Risk Contagion Model",
        "专题: 洪水风险与保险ABM模型": "Special Topic: Flood Risk & Insurance ABM Model",
        "专题: 欧元区经济危机AB-SFC模型": "Special Topic: Eurozone Economic Crisis AB-SFC Model",
        "专题: 粮食市场政策仿真模型": "Special Topic: Grain Market Policy Simulation Model"
    }
}

MODEL_CATALOG = {
    "6.2 农民作物选择模型": {
        "file": "6_2.py",
        "description_zh": "模拟农户在不同风险偏好下的作物选择行为，探索社会学习机制",
        "description_en": "Simulate farmers' crop selection behavior under different risk preferences, exploring social learning mechanisms",
        "chapter_zh": "第6章",
        "chapter_en": "Chapter 6",
        "keywords_zh": ["风险偏好", "社会学习", "作物选择", "涌现"],
        "keywords_en": ["Risk Preference", "Social Learning", "Crop Selection", "Emergence"],
        "theory_zh": """
### 一、理论基础

**1. 行为经济学 - 风险偏好异质性**
- 农户并非同质的"代表性个体"，而是具有不同风险承受能力的异质主体
- 风险厌恶型农户倾向选择低风险低收益的传统作物
- 风险偏好型农户愿意尝试高风险高收益的新作物

**2. 社会学习理论 - 邻居观察与模仿**
- 农户通过观察邻居的作物选择和收益情况进行学习
- 模仿机制：成功的邻居策略更容易被采纳
- 可能导致"羊群效应"或区域作物趋同

**3. 复杂系统理论 - 微观涌现宏观**
- 宏观作物分布模式由微观个体决策聚合产生
- 简单规则 → 复杂行为涌现

### 二、模型架构

```
┌─────────────────────────────────────────────────────────┐
│                    AgriculturalModel                     │
│                     (20×20 网格空间)                      │
├─────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │ Farmer  │  │ Farmer  │  │ Farmer  │  │  ...    │    │
│  │ 传统作物 │  │ 新作物  │  │ 多样化  │  │ (100个) │    │
│  └────┬────┘  └────┬────┘  └────┬────┘  └─────────┘    │
│       │            │            │                       │
│       └────────────┴────────────┘                       │
│              Moore邻域交互 (8邻居)                       │
└─────────────────────────────────────────────────────────┘
```

### 三、智能体决策规则

| 风险承受能力 | 作物选择 | 收益特征 |
|-------------|---------|----------|
| > 0.7 (高)  | 新作物   | 高收益高风险 (μ=10, σ=20) |
| 0.3-0.7 (中) | 多样化  | 中等收益中风险 (μ=8, σ=10) |
| < 0.3 (低)  | 传统作物 | 低收益低风险 (μ=5, σ=5) |

**社会学习机制**: 每期有30%概率模仿随机邻居的作物选择

### 四、预期结果

1. **作物多样性**: 三种作物共存，不会完全趋同
2. **空间聚类**: 相似作物类型在空间上形成聚集
3. **财富分化**: 高风险农户财富波动大，可能分化为富裕/贫困两极
""",
        "theory_en": """
### I. Theoretical Foundation

**1. Behavioral Economics - Risk Preference Heterogeneity**
- Farmers are not homogeneous "representative agents" but heterogeneous agents with different risk tolerance
- Risk-averse farmers tend to choose traditional crops with low risk and low returns
- Risk-preferring farmers are willing to try new crops with high risk and high returns

**2. Social Learning Theory - Neighbor Observation and Imitation**
- Farmers learn by observing neighbors' crop choices and returns
- Imitation mechanism: Successful neighbor strategies are more likely to be adopted
- May lead to "herding effect" or regional crop convergence

**3. Complex Systems Theory - Micro to Macro Emergence**
- Macro crop distribution patterns emerge from micro individual decisions
- Simple rules → Complex behavior emergence

### II. Model Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    AgriculturalModel                     │
│                     (20×20 Grid Space)                   │
├─────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │ Farmer  │  │ Farmer  │  │ Farmer  │  │  ...    │    │
│  │ Traditional│  │ New Crop│  │ Diversified│  │ (100) │    │
│  └────┬────┘  └────┬────┘  └────┬────┘  └─────────┘    │
│       │            │            │                       │
│       └────────────┴────────────┘                       │
│              Moore Neighborhood (8 neighbors)            │
└─────────────────────────────────────────────────────────┘
```

### III. Agent Decision Rules

| Risk Tolerance | Crop Choice | Return Characteristics |
|----------------|-------------|----------------------|
| > 0.7 (High)   | New Crop    | High return high risk (μ=10, σ=20) |
| 0.3-0.7 (Medium)| Diversified| Medium return medium risk (μ=8, σ=10) |
| < 0.3 (Low)    | Traditional | Low return low risk (μ=5, σ=5) |

**Social Learning Mechanism**: Each step has 30% probability to imitate a random neighbor's crop choice

### IV. Expected Results

1. **Crop Diversity**: Three crops coexist without complete convergence
2. **Spatial Clustering**: Similar crop types form spatial clusters
3. **Wealth Differentiation**: High-risk farmers have large wealth fluctuations, potentially polarizing into rich/poor
"""
    },
    "7.1 农民决策与环境影响模型": {
        "file": "7_1_2.py",
        "description": "引入环境意识和碳价格政策，分析农户决策与环境影响的关系",
        "chapter": "第7章",
        "keywords": ["环境政策", "碳价格", "可持续农业"],
        "theory": """
### 一、理论基础

**1. 环境经济学 - 外部性内部化**
- 传统农业的碳排放是负外部性
- 碳价格机制将外部成本内部化到农户决策中
- 庇古税理论：通过碳税使私人成本=社会成本

**2. 环境心理学 - 环境意识异质性**
- 农户环境意识(0-1)影响其决策权重
- 高环境意识农户：即使经济收益略低，也愿意选择环保作物
- 低环境意识农户：纯粹基于经济收益决策

### 二、模型架构

```
政策层面                    碳价格 (元/吨CO2)
    │                            │
    ▼                            ▼
┌─────────────────────────────────────────────────┐
│                   EnvFarmer                      │
│  ┌──────────────┐      ┌──────────────┐         │
│  │   传统作物    │      │   环保作物    │         │
│  │ 高产+高碳排放  │  vs  │ 中产+低碳排放  │         │
│  │ 碳成本=排放×碳价│     │ +环境意识加成  │         │
│  └──────────────┘      └──────────────┘         │
└─────────────────────────────────────────────────┘
```

### 三、决策函数

**传统作物净收益**:
```
π_传统 = 土地面积 × 500 - 化肥成本 × 土地面积 - 碳排放 × 碳价格
```

**环保作物净收益**:
```
π_环保 = 土地面积 × 400 - 碳排放 × 碳价格 + 环境意识 × 100
```

**转换规则**:
- 传统→环保: 环境意识 > 0.6 且 碳价格 > 30 时，有10%概率转换
- 环保→传统: 财富 < 50 时，有5%概率退回传统

### 四、政策实验设计

| 碳价格 (元/吨) | 预期环保比例 | 政策效果 |
|---------------|-------------|----------|
| 0             | ~30%        | 仅靠自发环境意识 |
| 50            | ~45%        | 温和政策效果 |
| 100           | ~60%        | 显著政策效果 |
| 200           | ~75%        | 强政策驱动 |
"""
    },
    "7.2 灌溉与水资源管理模型": {
        "file": "test_irrigation_model.py",
        "description": "模拟农民在不同水资源条件下的灌溉决策与策略选择",
        "chapter": "第7章",
        "keywords": ["水资源", "灌溉策略", "水分胁迫", "干旱适应"],
        "theory": """
### 一、理论基础

**1. 农业水文学 - 水分胁迫指数**
- 水分胁迫 = (蒸散量 - 有效降雨) / 最大亏缺
- 取值范围: 0(无胁迫) ~ 1(严重干旱)
- 胁迫程度直接影响作物产量

**2. 农户决策理论 - 成本-收益权衡**
- 灌溉决策需权衡: 灌溉成本 vs 产量损失
- 到水源距离影响灌溉成本
- 财富约束限制灌溉投入能力

### 二、水资源情景设计

| 情景 | 降雨均值 | 降雨标准差 | 典型地区 |
|------|---------|-----------|----------|
| 充足水资源 | 120mm | 20mm | 南方水稻区 |
| 中等缺水 | 80mm | 25mm | 华北平原 |
| 严重缺水 | 50mm | 30mm | 西北旱区 |

### 三、灌溉策略决策树

```
                水分胁迫指数
                    │
        ┌───────────┼───────────┐
        │           │           │
    < 0.3       0.3-0.6       > 0.6
        │           │           │
        ▼           ▼           ▼
      雨养      补充灌溉     充分灌溉?
    (无成本)    (中成本)    (检查财富)
                               │
                    ┌──────────┴──────────┐
                    │                      │
              财富 > 2×成本           财富不足
                    │                      │
                    ▼                      ▼
               充分灌溉              降级为补充灌溉
```

### 四、产量-策略关系

| 灌溉策略 | 产量系数 | 胁迫惩罚 | 成本 |
|---------|---------|---------|------|
| 充分灌溉 | 800 kg/亩 | ×(1-0.2×胁迫) | 1.5×基础成本 |
| 补充灌溉 | 600 kg/亩 | ×(1-0.4×胁迫) | 0.8×基础成本 |
| 雨养 | 400 kg/亩 | ×(1-0.7×胁迫) | 0 |

### 五、预期涌现模式

1. **策略分化**: 距水源近+富裕农户→充分灌溉；远+贫困→雨养
2. **财富马太效应**: 充分灌溉农户收益稳定，财富积累快
3. **干旱脆弱性**: 严重缺水情景下，雨养农户面临破产风险
"""
    },
    "8.1 农业保险采纳模型": {
        "file": "test_insurance_model.py",
        "description": "基于期望效用理论模拟农民保险购买决策与理赔过程",
        "chapter": "第8章",
        "keywords": ["农业保险", "风险管理", "政府补贴", "理赔"],
        "theory": """
### 一、理论基础

**1. 期望效用理论 (von Neumann-Morgenstern)**

决策准则: 最大化期望效用 E[U(W)]

```
E[U(无保险)] = (1-p)×U(W+收益) + p×U(W-损失)
E[U(有保险)] = (1-p)×U(W+收益-保费) + p×U(W-保费+赔付)
```

若 E[U(有保险)] > E[U(无保险)]，则购买保险

**2. 效用函数类型 (CRRA族)**

| 风险类型 | 效用函数 | CRRA系数 |
|---------|---------|----------|
| 风险中性 | U(W) = W | γ = 0 |
| 中等风险厌恶 | U(W) = √W | γ = 0.5 |
| 高度风险厌恶 | U(W) = ln(W) | γ = 1 |

### 二、保险产品设计

```
┌─────────────────────────────────────────────────────────┐
│                    保险合约结构                          │
├─────────────────────────────────────────────────────────┤
│  保险金额 = 预期产值 × 保障水平                          │
│  保费 = 保险金额 × 保费率 × (1 - 政府补贴率)             │
│  赔付 = min(实际损失, 保险金额) × 赔付比例(70%)          │
└─────────────────────────────────────────────────────────┘
```

### 三、政府补贴机制

| 补贴率 | 农户实缴保费 | 预期参保率 |
|--------|-------------|------------|
| 0%     | 全额自付    | ~20%       |
| 30%    | 70%        | ~40%       |
| 50%    | 50%        | ~60%       |
| 80%    | 20%        | ~85%       |

### 四、模型时序流程

```
每个时间步:
  1. 评估是否购买保险 (期望效用比较)
  2. 支付保费 (如已购买)
  3. 抽取灾害事件 (概率=disaster_prob)
  4. 若发生灾害:
     - 计算损失 = 产值 × 随机损失率(30%-80%)
     - 若有保险: 获得赔付，净损失 = 损失 - 赔付
     - 若无保险: 承担全部损失
  5. 若无灾害:
     - 获得正常经营收入
  6. 更新财富
```

### 五、预期结果

1. **逆向选择**: 高风险暴露农户更倾向购买保险
2. **财富保护**: 参保农户财富波动显著小于未参保
3. **补贴敏感性**: 参保率对补贴率高度敏感
"""
    },
    "8.2 农村信贷风险评估模型": {
        "file": "test_rural_credit_model.py",
        "description": "建模农民贷款申请、银行信贷决策与违约风险评估",
        "chapter": "第8章",
        "keywords": ["农村信贷", "信用评级", "违约风险", "贷款审批"],
        "theory": """
### 一、理论基础

**1. 信用风险理论**
- 违约概率(PD): 借款人无法履约的概率
- 违约损失率(LGD): 违约后银行损失占贷款比例
- 风险暴露(EAD): 违约时的贷款余额

**2. 信息不对称问题**
- 逆向选择: 高风险借款人更积极申请贷款
- 道德风险: 获得贷款后可能改变行为
- 银行通过信用评分缓解信息不对称

### 二、信用评分模型

**违约风险计算公式**:
```
PD = 0.4×债务收入比 + 0.4×(1-信用历史) + 0.2×(1-财富/200)
```

| 变量 | 权重 | 说明 |
|------|------|------|
| 债务收入比 | 40% | 债务/年收入，反映偿债压力 |
| 信用历史 | 40% | 0-1，历史还款记录 |
| 财富水平 | 20% | 抵押品价值代理变量 |

### 三、贷款审批决策

```
                  贷款申请
                     │
                     ▼
              信用评分 ≥ 阈值?
                     │
          ┌──────────┴──────────┐
          │                      │
         Yes                    No
          │                      │
          ▼                      ▼
    批准贷款                拒绝贷款
    利率=基准+风险溢价
```

**风险定价公式**:
```
利率 = 基准利率 + (1 - 信用评分) × 5%
```

### 四、农户收入冲击机制

模拟农业生产的固有风险:
- 每期12%概率遭遇收入冲击 (气候、病虫害、市场波动)
- 冲击强度: 收入下降15%-50%
- 财富同时受损(下降5%-30%)

### 五、违约与信用更新

**违约条件**: 还款额 > 当前财富 且 随机数 < 0.3

**信用更新规则**:
- 违约: 信用分 ×= 0.7 (惩罚)
- 正常还清: 信用分 ×= 1.05 (奖励，上限1.0)

### 六、预期涌现现象

1. **信贷配给**: 低信用农户被排斥在正规金融外
2. **违约聚集**: 收入冲击触发违约连锁
3. **信用分化**: 好农户越来越好，差农户恶性循环
"""
    },
    "10.2 农业技术扩散模型": {
        "file": "test_technology_diffusion_model.py",
        "description": "基于Rogers创新扩散理论的技术采用ABM模型，呈现S型扩散曲线",
        "chapter": "第10章",
        "keywords": ["技术扩散", "Rogers理论", "S型曲线", "创新采用"],
        "theory": """
### 一、理论基础

**1. Rogers创新扩散理论 (1962)**

采用者分类:

| 类型 | 比例 | 特征 | 创新性 |
|------|------|------|--------|
| 创新者 | 2.5% | 冒险、资源丰富 | > 0.95 |
| 早期采用者 | 13.5% | 意见领袖 | 0.8-0.95 |
| 早期多数 | 34% | 深思熟虑 | 0.5-0.8 |
| 晚期多数 | 34% | 持怀疑态度 | 0.2-0.5 |
| 落后者 | 16% | 传统导向 | < 0.2 |

**2. S型扩散曲线**

```
采用率
  │
1 ├──────────────────────╭───────
  │                   ╭──╯
  │                ╭──╯
  │             ╭──╯
  │          ╭──╯
  │       ╭──╯
0 ├───────╯─────────────────────→ 时间
       起飞点    拐点    饱和点
```

### 二、模型架构

```
┌─────────────────────────────────────────────────────────┐
│            TechnologyDiffusionModel (30×30网格)          │
├─────────────────────────────────────────────────────────┤
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                    │
│  │ 🟢  │  │ ⚪  │  │ ⚪  │  │ 🟢  │   🟢=已采用        │
│  │Agent│  │Agent│  │Agent│  │Agent│   ⚪=未采用        │
│  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘                    │
│     └────────┴────────┴────────┘                        │
│              Moore邻域 (8邻居)                           │
└─────────────────────────────────────────────────────────┘
```

### 三、采用概率计算

```
P(采用) = 0.5×邻居采用比例 + 0.3×创新性 + 0.2×财富因子
```

| 因素 | 权重 | 计算方式 |
|------|------|----------|
| 邻居影响 | 50% | 采用邻居数/总邻居数 |
| 个人创新性 | 30% | 0-1随机分配 |
| 财富能力 | 20% | min(1, 财富/200) |

### 四、空间扩散机制

1. **初始状态**: 5%随机分布的初始采用者(种子)
2. **局部传播**: 采用者周围的未采用者更容易采用
3. **空间聚类**: 采用者在空间上形成"斑块"
4. **波及扩散**: 类似涟漪向外扩展

### 五、预期结果

1. **S型曲线**: 累计采用率呈现典型S形
2. **钟形曲线**: 每期新采用者数量呈钟形分布
3. **空间聚类指数**: 采用者在空间上显著聚集
4. **采用者特征差异**: 早期vs晚期采用者的财富、创新性对比

完整代码已收录于附录E, 并同步保存在 test_technology_diffusion_model.py 中
"""
    },
    "12.2-12.5 银行风险传染模型": {
        "file": "test_banking_contagion_model.py",
        "description": "基于张亮(2017)理论框架的银行体系风险传染ABM模型",
        "chapter": "第12章",
        "keywords": ["银行风险", "网络传染", "系统性风险", "无标度网络"],
        "theory": """
### 一、理论基础

**1. 复杂网络理论**

| 网络类型 | 特征 | 金融含义 |
|---------|------|----------|
| 无标度网络 | 少数节点高度连接(hub) | 大银行系统重要性高 |
| 小世界网络 | 高聚类+短路径 | 风险快速传播 |
| 随机网络 | 均匀连接分布 | 风险分散 |

**2. 风险传染机制 (张亮, 2017)**

传染路径:
```
银行A违约 → 银行B对A有风险暴露 → B损失=暴露×LGD(60%)
          → B资本下降 → B资本充足率<8% → B压力上升
          → B违约概率增加 → 可能触发B违约 → 继续传染...
```

### 二、模型架构

```
                    ┌─────────────────┐
                    │   银行网络拓扑   │
                    │  (50家银行节点)  │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
     ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
     │ 无标度网络   │  │ 小世界网络   │  │ 随机网络    │
     │ BA模型 m=3  │  │ WS模型 k=6  │  │ ER模型 p=0.1│
     │ 幂律度分布   │  │ 高聚类短路径 │  │ 泊松度分布  │
     └─────────────┘  └─────────────┘  └─────────────┘
```

### 三、银行智能体设计

**资产负债表结构**:
```
资产端                          负债端
├── 贷款 (3-8倍资本)             ├── 存款 (5-10倍资本)
├── 同业资产                     ├── 同业负债
└── 资本 (初始100-10000)         └── 所有者权益
```

**关键指标**:
- 资本充足率 = 资本 / 总资产 (巴塞尔III要求≥8%)
- 杠杆率 = (存款+同业负债) / 资本
- 压力水平 (0-1): 累积风险指标

### 四、冲击情景设计

| 情景 | 初始冲击 | 预期结果 |
|------|---------|----------|
| 单个银行违约 | Bank_0违约 | 局部传染，系统基本稳定 |
| 多银行同时违约 | Bank_0,1,2违约 | 中等传染，稳定性下降 |
| 宏观经济冲击 | 全体压力+0~20% | 广泛但温和的影响 |

### 五、传染动力学

每个时间步:
```
1. 识别已违约银行集合 D
2. 对每个未违约银行 i:
   a. 检查邻居是否在D中
   b. 若有: 损失 = 同业资产 × 30%
           资本 -= 损失
           压力 += 0.15
   c. 检查资本充足率
   d. 若 < 8%: 压力 += 0.1
   e. 若 压力 > 0.5 且 随机 < 0.3: 违约
3. 收集系统指标
```

### 六、系统性风险指标

1. **系统稳定性** = 1 - 违约银行数/总银行数
2. **传染规模** = 因传染而违约的银行数
3. **平均压力水平** = Σ压力 / N
4. **网络聚类系数** = 反映银行间关联紧密度

### 七、实验结论（定性+定量）

**温和冲击下的稳健性**:
- 无标度网络: 稳定性~98%，单一违约不扩散
- 小世界网络: 稳定性~98%，局部传染可控
- 随机网络: 稳定性~100%，宏观冲击分散

**组合强冲击**: 当网络集中度高+冲击强度大时，系统性风险显著上升

---

### 八、扩展实验方向（基于张亮(2017)论文拓展）

本节基于banking_model_extensions.py中的六大扩展方向，支持政策评估与稳健性检验。

#### 方向1：政策干预实验
| 实验 | 机制 | 关键参数 |
|------|------|----------|
| 央行流动性救助 | 压力>0.7时注入资本 | 救助强度(10%/20%/30%) |
| 逆周期资本缓冲 | 风险指数>0.4时提高CAR | 触发阈值 |
| 同业拆借限额 | 限制单一暴露≤5%资本 | 暴露上限 |
| SIFI附加资本 | 中心性前20%银行额外+2%资本 | 附加资本率 |

#### 方向2：网络结构敏感性
- **网络密度扫描**: BA网络m=[2,3,4,5]，观察密度与传染关系
- **小世界重连概率**: WS网络p=[0.1,0.3,0.5,0.7]
- **核心-外围网络**: 10家大银行(核心)+40家小银行(外围)

#### 方向3：冲击场景丰富化
- **多点同时冲击**: 3-5家分散银行同时违约
- **渐进式冲击**: 存款每步-2%，持续10步
- **反复冲击**: 第1步冲击→平稳10步→第15步二次冲击
- **非对称冲击**: 仅对资本<中位数的小银行施加冲击

#### 方向4：参数敏感性检验
- **违约损失率(LGD)扫描**: [40%, 50%, 60%, 70%, 80%]
- **资本充足率阈值**: [6%, 8%, 10%, 12%]
- **压力累积速率**: [0.05, 0.1, 0.15, 0.2]

#### 方向5：蒙特卡洛稳健性检验
- 每个场景重复运行N次(不同随机种子)
- 计算违约率均值、标准差、95%置信区间
- 绘制概率分布直方图

#### 方向6：传染路径可视化
- 节点颜色映射: 红色=违约，橙色=高压力，蓝色=健康
- 节点大小映射: 资本规模
- 识别"超级传播者"和关键传染路径

---

完整代码已收录于附录E, 并同步保存在 test_banking_contagion_model.py 和 banking_model_extensions.py 中
"""
    },
    "专题: 洪水风险与保险ABM模型": {
        "file": "flood_insurance_abm_model.py",
        "description": "基于Dubbelboer(2017)的英国Flood Re再保险政策评估模型，模拟气候变化下的洪水风险管理",
        "chapter": "专题课",
        "keywords": ["洪水保险", "Flood Re", "气候变化", "PLPMs", "SUDS"],
        "theory": """
### 一、理论基础

**1. Flood Re再保险机制**
- 英国2016年建立的行业联合再保险计划
- 通过行业附加税(£10.50/保单)补贴高风险区域
- 2009年前房产可享受固定保费(不反映实际风险)

**2. 行为经济学基础**
- 可得性启发: 经历洪水后风险感知×1.2
- 乐观偏差: 风险感知~Beta(2,5)，均值≈0.29
- PLPMs投资: 灾后响应概率34%，主动预防仅1%

### 二、智能体类型

| 智能体 | 决策行为 |
|---------|----------|
| 居民(1000) | 保险购买、PLPMs投资、洪水应对 |
| 保险公司 | Flood Re资金池管理、索赔处理 |
| 地方政府 | SUDS投资(BCR≥5)、开发审批 |
| 开发商 | 项目提案(60%在洪泛区) |

### 三、气候情景

| 情景 | 高风险区洪水概率 | 特征 |
|------|----------------|------|
| 基线(1961-1990) | 10%/年 | 历史水平 |
| 2030s | 15%/年 | ×1.5倍增 |
| 2050s | 20%/年 | ×2倍增 |

### 四、关键机制

**保险购买决策**:
```
U_with = -premium                (确定性损失)
U_without = -expected_loss × risk_perception
if U_with > U_without: 购买保险
```

**深度-损失曲线**:
| 洪水深度 | 损失比例 |
|---------|----------|
| <0.3m | 10% |
| 0.3-0.6m | 30% |
| 0.6-1.0m | 50% |
| >1.0m | 70% |

### 五、预期结果

1. **Flood Re可持续性**: 基线情景资金池保持正值，2050s情景可能耗竭
2. **PLPMs采纳率**: 气候变化情景下显著上升
3. **财富分化**: 高风险区无保险居民财富损失最大
"""
    },
    "专题: 欧元区经济危机AB-SFC模型": {
        "file": "euro_crisis_app.py",
        "description": "基于存量-流量一致(SFC)框架的欧元区经济危机传导与政策干预ABM模型",
        "chapter": "专题课",
        "keywords": ["欧傺危机", "AB-SFC", "泰勒规则", "OMT", "信贷紧缩"],
        "theory": """
### 一、理论基础

**1. AB-SFC方法论**
- 智能体建模(ABM): 异质性主体、有限理性、微观涌现
- 存量流量一致(SFC): Copeland四重记账、部门平衡

**2. 危机传导机制**
- 2008金融危机: 需求冲击→信贷紧缩→GDP∙0.9
- 2010欧傺危机: 主权利差+500bp→OMT干预
- 2021通胀冲击: 工资成本→泰勒规则响应

### 二、模型架构

**空间结构**: 核心国(德/法) vs 外围国(希/葡)

| 智能体 | 数量 | 决策行为 |
|---------|------|----------|
| 企业 | 200家 | 生产/定价/投资/雇佣/融资 |
| 家庭 | 100户 | 消费/储蓄/资产组合 |
| 银行 | 2家 | 贷款审批/信贷标准调整 |
| 政府 | 2个 | 税收/支出/債务管理 |
| 央行 | 1个 | 泰勒规则/QE/OMT |

### 三、V7核心修复

**信贷紧缩机制**:
```
危机前: lending_standard=1.0 → 有效杠杆=10
危机后: lending_standard=1.5 → 有效杠杆=6.67
效果: 企业融资受限→投资↓→产出↓→GDP∙9%
```

**信心持久冲击**:
- 恢复速度: 0.05→0.01
- 冲击幅度: 0.20→0.40
- 效果: 100期后信心仅恢复63%，L型衰退

### 四、危机情景

| 情景 | 冲击内容 | 预期影响 |
|------|---------|----------|
| 基准 | 无冲击 | 稳定增长 |
| 2008金融危机 | 需求-30% | GDP∙9%，失业率12% |
| 2010欧傺危机 | 利差+500bp | 債务累积加速 |
| 2021通胀冲击 | 工资+6% | 泰勒规则响应 |

### 五、政策工具

- **OMT(直接货币交易)**: 强制利差≤25bp
- **泰勒规则**: i = r* + 1.5×(π-π*) + 0.5×y_gap
- **QE**: 量化宽松降低长端利率

### 六、预期结果

1. 金融危机: 信贷紧缩传导至实体经济，GDP下降~10%
2. 欧傺危机: OMT有效压缩利差490bp
3. 核心国 vs 外围国: 危机影响不对称
"""
    },
    "专题: 粮食市场政策仿真模型": {
        "file": "grain_market_mvp_model.py",
        "description": "基于2024年真实数据校准的粮食市场ABM模型，模拟关税、保险、信贷、农发行收购贷款等政策对粮食自给率和农户福利的影响",
        "chapter": "专题课",
        "keywords": ["粮食安全", "关税冲击", "大豆自给率", "农业保险", "银保联动", "农发行", "农地流转"],
        "theory": """
### 一、模型概述与校准数据来源

本模型基于2023-2024年中国官方统计数据进行参数校准：

| 数据类别 | 来源 | 校准值 |
|---------|------|--------|
| 粮食产量 | 国家统计局2024年12月13日公告 | 总产量7.065亿吨 |
| 主粮单产 | 稻谷+小麦+玉米加权平均 | **6500 kg/ha** |
| 大豆单产 | 国家统计局2024年 | **1991 kg/ha** |
| 主粮国内价 | 国家统计局流通领域价格监测(2024.10) | **2650元/吨** |
| 大豆国内价 | 同上 | **4200元/吨** |
| 主粮国际价 | 农业农村部国际市场周报(2024.11) | 1700元/吨 |
| 大豆国际价 | 同上 | 2960元/吨 |
| 谷物自给率 | 国家粮食和物资储备局 | **95%** |
| 大豆自给率 | 海关总署进口数据 | **约15%** |
| 三大作物保险覆盖率 | 银保监会 | **82%** |
| 中央保险补贴(中西部) | 财政部财金〔2023〕59号 | **45%** |
| 普惠贷款利率 | 中国人民银行普惠金融报告 | **4.13%** |
| 农发行收购贷款利率 | 农业发展银行官网 | **3.25%** |
| 小农户占比 | 全国人大农业法执法检查报告(2024) | **98%** |
| 农村人均可支配收入 | 国家统计局2025年1月17日 | **23119元** |

### 二、主体类型与行为规则

**1. 农户主体 (FarmerAgent)**
- 属性: 土地面积、财富、风险厌恶系数、作物结构
- 决策: 种植结构(惯性调整±10%)、保险购买、信贷申请
- 生产函数: 产量 = 面积 × 单产 × (1-灾损率)

**2. 政府主体 (GovernmentAgent)**
- 政策工具: 关税、保险补贴(中央45%+省25%+县10%)、信贷贴息、生产补贴、托市价
- 财政预算约束与支出统计

**3. 保险公司 (InsuranceFirm)**
- 产品类型: 完全成本保险(4%)、收入保险(6%)、特色保险(3%)
- 理赔机制: 灾损超阈值时按保障水平赔付

**4. 农村银行 (RuralBank)**
- A/B/C分层信贷: A级(80%额度,+0.5%)、B级(50%,+2%)、C级(30%,+5%)
- 银保联动: 参保农户利率-10%，额度+20%

**5. 农发行 (PolicyBank)**
- 政策性收购贷款: 基准利率3.25%
- 按收购商类型差异化支持: 国企粮库(90%)>加工企业(72%)>贸易商(54%)

**6. 收购商 (DomesticBuyer)**
- 国企粮库: 托市价收购，政策导向
- 加工企业: 市场价+10%品质溢价
- 贸易商: 套利定价

### 三、核心传导链条

```
关税↑ → 进口成本↑ → 国内价格↑ → 大豆收益↑ → 面积扩张 → 产量↑ → 自给率↑

保险补贴↑ → 参保率↑ → 收入稳定 → 风险承担↑ → 扩大生产

银保联动: 保险→降低违约风险→提高贷款额度→扩大再生产

农发行收购贷款 → 稳定粮食收购 → 保障农户售粮 → 稳定种植预期
```

### 四、价格形成机制

国内价格 = 国际价格 × (1 + 关税率) × 供需调整系数

供需调整系数 = 1 + 0.5 × (1 - 供给/需求)

价格约束: 0.8×国际价 ≤ 国内价 ≤ 3×国际价

### 五、实验情景

| 情景 | 政策设定 | 预期效果 |
|------|---------|----------|
| 基线场景 | 关税1%，无金融模块 | 维持现状 |
| 关税冲击 | 第5期关税↑至15% | 短期价格上涨，中期自给率改善 |
| 保险扩面 | 补贴↑，农户自缴10% | 参保率提升至80%+ |
| 银保联动 | 保险+信贷协同 | 规模经营主体扩张 |
| 农发行支持 | 启用政策性收购贷款 | 稳定粮价，保障农户售粮 |
| 农地流转 | 每期2%小农户退出 | 规模化经营比例提升 |
| 全政策组合 | 关税15%+保险+信贷+农发行+流转 | 综合效果最优 |
"""
    },
}

# =============================================================================
# 侧边栏 - 模型选择与参数配置
# =============================================================================
with st.sidebar:
    st.header(t('experiment_config'))
    
    # 语言选择
    st.subheader("🌐 " + t('language'))
    selected_language = st.radio(
        "language_selector",
        ["中文", "English"],
        index=0 if st.session_state.language == "中文" else 1,
        horizontal=True,
        label_visibility="collapsed"
    )
    if selected_language != st.session_state.language:
        st.session_state.language = selected_language
        st.rerun()
    
    st.markdown("---")
    
    # 字体大小选择
    st.subheader(t('display_settings'))
    selected_font = st.selectbox(
        t('font_size_label'),
        list(FONT_SIZE_MAP.keys()),
        index=list(FONT_SIZE_MAP.keys()).index(st.session_state.font_size),
        help="选择页面显示的字体大小，默认四号字体"
    )
    if selected_font != st.session_state.font_size:
        st.session_state.font_size = selected_font
        st.rerun()
    
    st.markdown("---")
    
    # 模型选择
    st.subheader(t('select_model'))
    
    # 获取当前语言的模型名称列表
    lang = st.session_state.language
    model_display_names = [MODEL_NAMES[lang][key] for key in MODEL_CATALOG.keys()]
    model_name_to_key = {MODEL_NAMES[lang][key]: key for key in MODEL_CATALOG.keys()}
    
    selected_model_display = st.selectbox(
        "请选择要运行的模型 / Select model to run",
        model_display_names,
        help="选择讲义中的ABM实验模型 / Select ABM experiment model from lecture"
    )
    
    # 将显示名称映射回原始键
    selected_model = model_name_to_key[selected_model_display]
    
    # 显示模型信息
    model_info = MODEL_CATALOG[selected_model]
    chapter = model_info.get('chapter_zh', model_info.get('chapter', ''))
    if st.session_state.language == "English":
        chapter = model_info.get('chapter_en', chapter)
        description = model_info.get('description_en', model_info.get('description', ''))
        keywords = model_info.get('keywords_en', model_info.get('keywords', []))
    else:
        description = model_info.get('description_zh', model_info.get('description', ''))
        keywords = model_info.get('keywords_zh', model_info.get('keywords', []))
    
    st.info(f"**{chapter}**\n\n{description}")
    st.caption(f"{t('keywords')}: {', '.join(keywords)}")
    
    st.markdown("---")
    
    # 通用参数
    st.subheader(t('common_params'))
    
    n_steps = st.slider(
        t('simulation_steps'),
        min_value=10, max_value=200, value=50, step=10,
        help="模型运行的时间步数 / Number of simulation steps"
    )
    
    random_seed = st.number_input(
        t('random_seed'),
        min_value=1, max_value=9999, value=42,
        help="设置相同的种子可复现结果 / Set same seed for reproducibility"
    )
    
    st.markdown("---")
    
    # 模型特定参数
    st.subheader(t('model_params'))
    
    # ========== 6.2 农民作物选择模型参数 ==========
    if selected_model == "6.2 农民作物选择模型":
        n_farmers_62 = st.slider(t("n_farmers"), 50, 500, 100, 10)
        grid_size_62 = st.slider(t("grid_size"), 10, 50, 20, 5)
        social_learning_prob = st.slider(t("social_learning_prob"), 0.0, 1.0, 0.3, 0.05)
        
        model_params = {
            "n_farmers": n_farmers_62,
            "grid_size": grid_size_62,
            "social_learning_prob": social_learning_prob
        }
    
    # ========== 7.1 农民决策与环境影响模型参数 ==========
    elif selected_model == "7.1 农民决策与环境影响模型":
        n_farmers_71 = st.slider(t("n_farmers"), 50, 300, 100, 10)
        carbon_price = st.slider(t("carbon_price"), 0, 200, 50, 10)
        fertilizer_cost = st.slider(t("fertilizer_cost"), 100, 1000, 300, 50)
        
        model_params = {
            "n_farmers": n_farmers_71,
            "carbon_price": carbon_price,
            "fertilizer_cost": fertilizer_cost
        }
    
    # ========== 7.2 灌溉与水资源管理模型参数 ==========
    elif selected_model == "7.2 灌溉与水资源管理模型":
        n_farmers_72 = st.slider(t("n_farmers"), 30, 200, 80, 10)
        water_scenario = st.selectbox(
            t("water_scenario"),
            [t("adequate_water"), t("moderate_water_shortage"), t("severe_water_shortage")],
            help=t("water_scenario")
        )
        irrigation_cost_factor = st.slider(t("irrigation_cost_factor"), 0.5, 3.0, 1.0, 0.1)
        
        model_params = {
            "n_farmers": n_farmers_72,
            "water_scenario": water_scenario,
            "irrigation_cost_factor": irrigation_cost_factor
        }
    
    # ========== 8.1 农业保险采纳模型参数 ==========
    elif selected_model == "8.1 农业保险采纳模型":
        n_farmers_81 = st.slider(t("n_farmers"), 50, 300, 100, 10)
        premium_rate = st.slider(t("premium_rate"), 1.0, 10.0, 4.0, 0.5) / 100
        subsidy_rate = st.slider(t("subsidy_rate"), 0, 80, 50, 5) / 100
        disaster_prob = st.slider(t("disaster_prob"), 5, 50, 15, 5) / 100
        
        model_params = {
            "n_farmers": n_farmers_81,
            "premium_rate": premium_rate,
            "subsidy_rate": subsidy_rate,
            "disaster_prob": disaster_prob
        }
    
    # ========== 8.2 农村信贷风险评估模型参数 ==========
    elif selected_model == "8.2 农村信贷风险评估模型":
        n_farmers_82 = st.slider(t("n_farmers"), 50, 300, 100, 10)
        base_interest_rate = st.slider(t("base_interest_rate"), 3.0, 15.0, 6.0, 0.5) / 100
        credit_threshold = st.slider(t("credit_threshold"), 0.3, 0.8, 0.5, 0.05)
        
        model_params = {
            "n_farmers": n_farmers_82,
            "base_interest_rate": base_interest_rate,
            "credit_threshold": credit_threshold
        }
    
    # ========== 10.2 农业技术扩散模型参数 ==========
    elif selected_model == "10.2 农业技术扩散模型":
        n_agents_102 = st.slider(t("n_farmers"), 100, 1000, 400, 50)
        grid_size_102 = st.slider(t("grid_size"), 15, 50, 30, 5)
        initial_adopters = st.slider(t("initial_adopters"), 1, 20, 5, 1) / 100
        
        model_params = {
            "n_agents": n_agents_102,
            "grid_size": grid_size_102,
            "initial_adopters": initial_adopters
        }
    
    # ========== 12.2-12.5 银行风险传染模型参数 ==========
    elif selected_model == "12.2-12.5 银行风险传染模型":
        st.markdown(f"#### {t('basic_params')}")
        n_banks = st.slider(t("n_banks"), 20, 100, 50, 10)
        network_type = st.selectbox(
            t("network_topology"),
            [t("scale_free_network"), t("small_world_network"), t("random_network")]
        )
        shock_scenario = st.selectbox(
            t("shock_scenario"),
            [t("single_bank_default"), t("multiple_banks_default"), t("distributed_shocks"), t("macroeconomic_shock"), t("progressive_shock")]
        )
        
        # 解析网络类型
        network_map = {
            t("scale_free_network"): "scale_free",
            t("small_world_network"): "small_world",
            t("random_network"): "random"
        }
        
        st.markdown("---")
        st.markdown(f"#### {t('extended_params')}")
        
        # 政策干预实验
        enable_bailout = st.checkbox(t("enable_bailout"), value=False, 
                                     help=t("enable_bailout"))
        bailout_intensity = st.slider(t("bailout_intensity"), 10, 50, 20, 5) / 100 if enable_bailout else 0.2
        
        # 网络结构敏感性
        network_density = st.slider(t("network_density"), 2, 6, 3, 1,
                                   help=t("network_density"))
        
        # 参数敏感性
        lgd = st.slider(t("lgd"), 30, 80, 60, 5) / 100
        car_threshold = st.slider(t("car_threshold"), 4, 12, 8, 1) / 100
        stress_increment = st.slider(t("stress_increment"), 0.05, 0.25, 0.15, 0.05)
        
        # 蒙特卡洛实验
        st.markdown("---")
        st.markdown(f"#### {t('robustness_test')}")
        enable_monte_carlo = st.checkbox(t("enable_monte_carlo"), value=False,
                                        help=t("enable_monte_carlo"))
        monte_carlo_runs = st.slider(t("monte_carlo_runs"), 10, 100, 30, 10) if enable_monte_carlo else 1
        
        # 可视化选项
        show_network_viz = st.checkbox(t("show_network_viz"), value=True)
        
        model_params = {
            "n_banks": n_banks,
            "network_type": network_map[network_type],
            "shock_scenario": shock_scenario,
            "enable_bailout": enable_bailout,
            "bailout_intensity": bailout_intensity,
            "network_density": network_density,
            "lgd": lgd,
            "car_threshold": car_threshold,
            "stress_increment": stress_increment,
            "enable_monte_carlo": enable_monte_carlo,
            "monte_carlo_runs": monte_carlo_runs,
            "show_network_viz": show_network_viz
        }
    
    # ========== 专题: 洪水风险与保险ABM模型参数 ==========
    elif selected_model == "专题: 洪水风险与保险ABM模型":
        n_residents_flood = st.slider("居民数量", 200, 2000, 1000, 100)
        climate_scenario = st.selectbox(
            "气候情景",
            ["基线情景 (baseline)", "2030s情景", "2050s情景"],
            help="模拟不同气候变化情景下的洪水风险"
        )
        
        # 解析气候情景
        climate_map = {
            "基线情景 (baseline)": "baseline",
            "2030s情景": "2030s",
            "2050s情景": "2050s"
        }
        
        model_params = {
            "n_residents": n_residents_flood,
            "climate_scenario": climate_map[climate_scenario]
        }
    
    # ========== 专题: 欧元区经济危机AB-SFC模型参数 ==========
    elif selected_model == "专题: 欧元区经济危机AB-SFC模型":
        n_firms_euro = st.slider("每国企业数", 50, 200, 100, 10)
        n_households_euro = st.slider("每国家庭数", 20, 100, 50, 10)
        crisis_scenario = st.selectbox(
            "危机情景",
            ["基准情景", "2008金融危机", "2010欧傺危机", "2021通胀冲击"],
            help="选择危机类型"
        )
        omt_enabled = st.checkbox("OMT政策干预", value=True, help="启用欧洲央行直接货币交易")
        
        # 解析危机情景
        crisis_map = {
            "基准情景": "baseline",
            "2008金融危机": "financial_crisis",
            "2010欧傺危机": "sovereign_crisis",
            "2021通胀冲击": "inflation_shock"
        }
        
        model_params = {
            "n_firms": n_firms_euro,
            "n_households": n_households_euro,
            "crisis_scenario": crisis_map[crisis_scenario],
            "omt_enabled": omt_enabled
        }
    
    # ========== 专题: 粮食市场政策仿真模型参数 ==========
    elif selected_model == "专题: 粮食市场政策仿真模型":
        st.subheader("🌾 粮食市场模型参数")
        
        # 检查完整模型是否可用
        if not GRAIN_MODEL_AVAILABLE:
            st.warning("⎡ 完整粮食市场模型未加载，使用简化版本")
        
        # 基础参数
        st.markdown("**📊 基础参数 (校准数据来源: 全国人大农业法执法检查报告2024)**")
        n_farmers_grain = st.select_slider(
            "代理数量 (Agent Count)",
            options=[100, 200, 500, 1000, 2000],
            value=500,
            help="每个代理代表约万户农户，共代表全国2.01亿确权农户"
        )
        
        # 显示校准信息
        if GRAIN_MODEL_AVAILABLE:
            st.caption(f"📌 校准值: 小农户占98% | 主粮单产6500kg/ha | 大豆单产1991kg/ha")
        
        # 政策模块开关
        st.markdown("**🔧 政策模块开关**")
        col_sw1, col_sw2 = st.columns(2)
        with col_sw1:
            enable_land_transfer = st.checkbox("开启农地流转", value=False, help="小农户退出农业，土地流转给规模经营主体")
            enable_insurance = st.checkbox("开启农业保险模块", value=True, help="启用农业保险功能")
        with col_sw2:
            enable_credit = st.checkbox("开启农村信贷模块", value=False, help="启用商业银行农户贷款")
            enable_policy_bank = st.checkbox("开启农发行收购贷款", value=False, help="启用政策性收购贷款")
        
        # 关税政策
        st.markdown("**📦 关税政策 (数据来源: 财政部2024年关税调整方案)**")
        enable_tariff_shock = st.checkbox("启用关税冲击", value=False, help="在指定时期提高大豆关税")
        
        col_tariff1, col_tariff2 = st.columns(2)
        with col_tariff1:
            initial_tariff_soybean = st.slider(
                "大豆初始关税率 (%)",
                min_value=0, max_value=10, value=1,
                help="校准值: 暂定税率0%，基线用 1%"
            )
        with col_tariff2:
            if enable_tariff_shock:
                tariff_shock_rate = st.slider(
                    "冲击后关税率 (%)",
                    min_value=5, max_value=30, value=15
                )
                tariff_shock_step = st.number_input("冲击时期", min_value=1, max_value=20, value=5)
            else:
                tariff_shock_rate = 15
                tariff_shock_step = 5
        
        # 保险政策
        if enable_insurance:
            st.markdown("**🛡️ 保险政策 (校准: 财政部财金〔2023〕59号)**")
            col_ins1, col_ins2, col_ins3 = st.columns(3)
            with col_ins1:
                insurance_central = st.slider("中央补贴(%)", 30, 55, 45, help="校准值: 中西部45%")
            with col_ins2:
                insurance_province = st.slider("省级补贴(%)", 15, 35, 25, help="校准值: ≥25%")
            with col_ins3:
                insurance_county = st.slider("县级补贴(%)", 5, 20, 10)
            insurance_farmer = 100 - insurance_central - insurance_province - insurance_county
            st.info(f"👨‍🌾 农户自缴比例: **{insurance_farmer}%** | 三大粮食作物保险覆盖率校准值: 82%")
        else:
            insurance_central, insurance_province, insurance_county = 45, 25, 10
            insurance_farmer = 20
        
        # 信贷政策
        if enable_credit:
            st.markdown("**🏦 信贷政策 (校准: 人行普惠金融报告2024)**")
            base_interest_rate = st.slider(
                "基准利率 (%)",
                min_value=3.0, max_value=6.0, value=4.13, step=0.1,
                help="校准值: 普惠小微贷款平均利4.13%"
            )
            enable_bank_insurance_linkage = st.checkbox("启用银保联动", value=True, help="参保农户利率-10%，额度+20%")
        else:
            base_interest_rate = 4.13
            enable_bank_insurance_linkage = True
        
        # 农发行政策
        if enable_policy_bank:
            st.markdown("**🏛️ 农发行收购贷款 (校准: 农业发展银行官网)**")
            policy_bank_rate = st.slider(
                "收购贷款基准利率 (%)",
                min_value=2.5, max_value=4.5, value=3.25, step=0.25,
                help="校准值: 粮棉油收购贷款约3.25%"
            )
            st.caption("国企粮库贷款比例90% > 加工企业72% > 贸易商54%")
        else:
            policy_bank_rate = 3.25
        
        # 生产补贴
        st.markdown("**💰 生产补贴 (元/公顷)**")
        col_sub1, col_sub2 = st.columns(2)
        with col_sub1:
            subsidy_staple = st.number_input("主粮补贴", 0, 500, 100, 50)
        with col_sub2:
            subsidy_soybean = st.number_input("大豆补贴", 0, 500, 200, 50)
        
        # 政策情景选择
        st.markdown("**🎯 政策情景**")
        policy_scenario = st.selectbox(
            "预设情景",
            ["自定义", "基线情景", "关税冲击(15%)", "保险扩面", "银保联动", "全政策组合"],
            help="选择预设情景或自定义参数"
        )
        
        # 根据情景调整参数
        if policy_scenario == "关税冲击(15%)":
            enable_tariff_shock = True
            tariff_shock_rate = 15
        elif policy_scenario == "保险扩面":
            enable_insurance = True
            insurance_central, insurance_province, insurance_county = 50, 30, 10
            insurance_farmer = 10
        elif policy_scenario == "银保联动":
            enable_insurance = True
            enable_credit = True
            enable_bank_insurance_linkage = True
        elif policy_scenario == "全政策组合":
            enable_tariff_shock = True
            tariff_shock_rate = 15
            enable_insurance = True
            enable_credit = True
            enable_policy_bank = True
            enable_land_transfer = True
        
        # 构建参数字典
        model_params = {
            "n_farmers": n_farmers_grain,
            "enable_land_transfer": enable_land_transfer,
            "enable_insurance": enable_insurance,
            "enable_credit": enable_credit,
            "enable_policy_bank": enable_policy_bank,
            "initial_tariff_soybean": initial_tariff_soybean / 100,
            "enable_tariff_shock": enable_tariff_shock,
            "tariff_shock_rate": tariff_shock_rate / 100,
            "tariff_shock_step": tariff_shock_step,
            "insurance_central": insurance_central / 100,
            "insurance_province": insurance_province / 100,
            "insurance_county": insurance_county / 100,
            "insurance_farmer": insurance_farmer / 100,
            "base_interest_rate": base_interest_rate / 100,
            "enable_bank_insurance_linkage": enable_bank_insurance_linkage,
            "policy_bank_rate": policy_bank_rate / 100,
            "subsidy_staple": subsidy_staple,
            "subsidy_soybean": subsidy_soybean,
            "policy_scenario": policy_scenario
        }
    
    else:
        model_params = {}
    
    st.markdown("---")
    
    # 运行按钮
    run_simulation = st.button(t('run_simulation'), type="primary")


# =============================================================================
# 模型运行函数
# =============================================================================

def run_crop_choice_model(params, n_steps, seed):
    """运行6.2农民作物选择模型"""
    random.seed(seed)
    np.random.seed(seed)
    
    class Farmer(Agent):
        def __init__(self, model):
            super().__init__(model)
            self.wealth = random.uniform(50, 150)
            self.risk_tolerance = random.random()
            if self.risk_tolerance > 0.7:
                self.crop_type = "新作物"
            elif self.risk_tolerance > 0.3:
                self.crop_type = "多样化"
            else:
                self.crop_type = "传统作物"
        
        def step(self):
            # 社会学习
            if random.random() < params["social_learning_prob"]:
                neighbors = self.model.grid.get_neighbors(
                    self.pos, moore=True, include_center=False
                )
                if neighbors:
                    neighbor = random.choice(list(neighbors))
                    self.crop_type = neighbor.crop_type
            
            # 更新财富
            if self.crop_type == "新作物":
                self.wealth += random.normalvariate(10, 20)
            elif self.crop_type == "多样化":
                self.wealth += random.normalvariate(8, 10)
            else:
                self.wealth += random.normalvariate(5, 5)
            self.wealth = max(10, self.wealth)
    
    class CropModel(Model):
        def __init__(self, n_farmers, grid_size):
            super().__init__()
            self.grid = MultiGrid(grid_size, grid_size, torus=True)
            self.current_step = 0
            
            for i in range(n_farmers):
                a = Farmer(self)
                x = random.randrange(grid_size)
                y = random.randrange(grid_size)
                self.grid.place_agent(a, (x, y))
            
            self.datacollector = DataCollector(
                model_reporters={
                    "传统作物比例": lambda m: sum(1 for a in m.agents if a.crop_type == "传统作物") / len(list(m.agents)),
                    "新作物比例": lambda m: sum(1 for a in m.agents if a.crop_type == "新作物") / len(list(m.agents)),
                    "多样化比例": lambda m: sum(1 for a in m.agents if a.crop_type == "多样化") / len(list(m.agents)),
                    "平均财富": lambda m: np.mean([a.wealth for a in m.agents])
                }
            )
        
        def step(self):
            self.current_step += 1
            self.agents.shuffle_do("step")
            self.datacollector.collect(self)
    
    model = CropModel(params["n_farmers"], params["grid_size"])
    
    progress = st.progress(0)
    for i in range(n_steps):
        model.step()
        progress.progress((i + 1) / n_steps)
    progress.empty()
    
    return model.datacollector.get_model_vars_dataframe()


def run_environmental_model(params, n_steps, seed):
    """运行7.1农民决策与环境影响模型"""
    random.seed(seed)
    np.random.seed(seed)
    
    class EnvFarmer(Agent):
        def __init__(self, model, carbon_price, fertilizer_cost):
            super().__init__(model)
            self.wealth = random.uniform(50, 200)
            self.env_awareness = random.random()
            self.land_area = random.uniform(1, 10)
            self.carbon_price = carbon_price
            self.fertilizer_cost = fertilizer_cost
            self.crop_type = "传统" if random.random() > 0.3 else "环保"
            self.carbon_emission = 0
        
        def step(self):
            # 决策逻辑
            if self.crop_type == "传统":
                profit = self.land_area * 500 - self.fertilizer_cost * self.land_area
                self.carbon_emission = self.land_area * 2.5
                carbon_cost = self.carbon_emission * self.carbon_price
                net_profit = profit - carbon_cost
            else:
                profit = self.land_area * 400
                self.carbon_emission = self.land_area * 0.8
                carbon_cost = self.carbon_emission * self.carbon_price
                net_profit = profit - carbon_cost + self.env_awareness * 100
            
            self.wealth += net_profit / 100
            
            # 决策转换
            if self.crop_type == "传统" and self.env_awareness > 0.6 and self.carbon_price > 30:
                if random.random() < 0.1:
                    self.crop_type = "环保"
            elif self.crop_type == "环保" and self.wealth < 50:
                if random.random() < 0.05:
                    self.crop_type = "传统"
    
    class EnvModel(Model):
        def __init__(self, n_farmers, carbon_price, fertilizer_cost):
            super().__init__()
            self.carbon_price = carbon_price
            for _ in range(n_farmers):
                EnvFarmer(self, carbon_price, fertilizer_cost)
            
            self.datacollector = DataCollector(
                model_reporters={
                    "环保作物比例": lambda m: sum(1 for a in m.agents if a.crop_type == "环保") / len(list(m.agents)),
                    "平均碳排放": lambda m: np.mean([a.carbon_emission for a in m.agents]),
                    "平均财富": lambda m: np.mean([a.wealth for a in m.agents])
                }
            )
        
        def step(self):
            self.agents.shuffle_do("step")
            self.datacollector.collect(self)
    
    model = EnvModel(params["n_farmers"], params["carbon_price"], params["fertilizer_cost"])
    
    progress = st.progress(0)
    for i in range(n_steps):
        model.step()
        progress.progress((i + 1) / n_steps)
    progress.empty()
    
    return model.datacollector.get_model_vars_dataframe()


def run_irrigation_model(params, n_steps, seed):
    """运行7.2灌溉与水资源管理模型"""
    random.seed(seed)
    np.random.seed(seed)
    
    # 水资源情景参数
    scenario_params = {
        "充足水资源": {"rainfall_mean": 120, "rainfall_std": 20},
        "中等缺水": {"rainfall_mean": 80, "rainfall_std": 25},
        "严重缺水": {"rainfall_mean": 50, "rainfall_std": 30}
    }
    
    scenario = scenario_params[params["water_scenario"]]
    
    class IrrigationFarmer(Agent):
        def __init__(self, model, cost_factor):
            super().__init__(model)
            self.wealth = random.uniform(50, 200)
            self.land_size = random.uniform(2, 15)
            self.water_distance = random.uniform(0.5, 5)
            self.irrigation_cost = self.water_distance * 20 * cost_factor
            self.strategy = "未决策"
            self.water_stress = 0
            self.crop_yield = 0
        
        def step(self):
            # 计算水分胁迫
            rainfall = max(0, random.normalvariate(scenario["rainfall_mean"], scenario["rainfall_std"]))
            evap = random.normalvariate(100, 15)
            deficit = max(0, evap - rainfall)
            self.water_stress = min(1.0, deficit / 100)
            
            # 灌溉决策
            if self.water_stress > 0.6:
                if self.wealth > self.irrigation_cost * 2:
                    self.strategy = "充分灌溉"
                else:
                    self.strategy = "补充灌溉"
            elif self.water_stress > 0.3:
                self.strategy = "补充灌溉"
            else:
                self.strategy = "雨养"
            
            # 产量计算
            if self.strategy == "充分灌溉":
                self.crop_yield = self.land_size * 800 * (1 - self.water_stress * 0.2)
                cost = self.irrigation_cost * 1.5
            elif self.strategy == "补充灌溉":
                self.crop_yield = self.land_size * 600 * (1 - self.water_stress * 0.4)
                cost = self.irrigation_cost * 0.8
            else:
                self.crop_yield = self.land_size * 400 * (1 - self.water_stress * 0.7)
                cost = 0
            
            income = self.crop_yield * 0.5 - cost
            self.wealth += income / 10
            self.wealth = max(10, self.wealth)
    
    class IrrigationModel(Model):
        def __init__(self, n_farmers, cost_factor):
            super().__init__()
            for _ in range(n_farmers):
                IrrigationFarmer(self, cost_factor)
            
            self.datacollector = DataCollector(
                model_reporters={
                    "充分灌溉比例": lambda m: sum(1 for a in m.agents if a.strategy == "充分灌溉") / len(list(m.agents)),
                    "补充灌溉比例": lambda m: sum(1 for a in m.agents if a.strategy == "补充灌溉") / len(list(m.agents)),
                    "雨养比例": lambda m: sum(1 for a in m.agents if a.strategy == "雨养") / len(list(m.agents)),
                    "平均水分胁迫": lambda m: np.mean([a.water_stress for a in m.agents]),
                    "平均产量": lambda m: np.mean([a.crop_yield for a in m.agents]),
                    "平均财富": lambda m: np.mean([a.wealth for a in m.agents])
                }
            )
        
        def step(self):
            self.agents.shuffle_do("step")
            self.datacollector.collect(self)
    
    model = IrrigationModel(params["n_farmers"], params["irrigation_cost_factor"])
    
    progress = st.progress(0)
    for i in range(n_steps):
        model.step()
        progress.progress((i + 1) / n_steps)
    progress.empty()
    
    return model.datacollector.get_model_vars_dataframe()


def run_insurance_model(params, n_steps, seed):
    """运行8.1农业保险采纳模型"""
    random.seed(seed)
    np.random.seed(seed)
    
    class InsuranceAgent(Agent):
        def __init__(self, model, premium_rate, subsidy_rate, disaster_prob):
            super().__init__(model)
            self.wealth = random.uniform(50, 300)
            self.farm_size = random.uniform(2, 20)
            self.risk_aversion = random.random()
            self.has_insurance = False
            self.premium_rate = premium_rate
            self.subsidy_rate = subsidy_rate
            self.disaster_prob = disaster_prob
            self.total_loss = 0
            self.total_payout = 0
        
        def step(self):
            expected_value = self.farm_size * 1000  # 预期产值
            premium = expected_value * self.premium_rate * (1 - self.subsidy_rate)
            
            # 购买决策（基于期望效用）
            if not self.has_insurance:
                # 无保险期望效用
                eu_no_ins = (1 - self.disaster_prob) * self._utility(self.wealth + expected_value * 0.3) + \
                           self.disaster_prob * self._utility(self.wealth - expected_value * 0.5)
                # 有保险期望效用
                eu_with_ins = (1 - self.disaster_prob) * self._utility(self.wealth + expected_value * 0.3 - premium) + \
                             self.disaster_prob * self._utility(self.wealth - premium)
                
                if eu_with_ins > eu_no_ins and self.wealth > premium * 2:
                    self.has_insurance = True
            
            # 灾害发生
            if random.random() < self.disaster_prob:
                loss = expected_value * random.uniform(0.3, 0.8)
                self.total_loss += loss
                if self.has_insurance:
                    payout = loss * 0.7
                    self.total_payout += payout
                    self.wealth -= (loss - payout)
                else:
                    self.wealth -= loss
            else:
                income = expected_value * 0.3
                if self.has_insurance:
                    income -= premium
                self.wealth += income
            
            self.wealth = max(10, self.wealth)
        
        def _utility(self, w):
            if w <= 0:
                return -1e10
            if self.risk_aversion < 0.33:
                return w
            elif self.risk_aversion < 0.67:
                return np.sqrt(w)
            else:
                return np.log(w)
    
    class InsuranceModel(Model):
        def __init__(self, n_farmers, premium_rate, subsidy_rate, disaster_prob):
            super().__init__()
            for _ in range(n_farmers):
                InsuranceAgent(self, premium_rate, subsidy_rate, disaster_prob)
            
            self.datacollector = DataCollector(
                model_reporters={
                    "参保率": lambda m: sum(1 for a in m.agents if a.has_insurance) / len(list(m.agents)),
                    "平均财富": lambda m: np.mean([a.wealth for a in m.agents]),
                    "累计损失": lambda m: sum(a.total_loss for a in m.agents),
                    "累计赔付": lambda m: sum(a.total_payout for a in m.agents)
                }
            )
        
        def step(self):
            self.agents.shuffle_do("step")
            self.datacollector.collect(self)
    
    model = InsuranceModel(
        params["n_farmers"], params["premium_rate"],
        params["subsidy_rate"], params["disaster_prob"]
    )
    
    progress = st.progress(0)
    for i in range(n_steps):
        model.step()
        progress.progress((i + 1) / n_steps)
    progress.empty()
    
    return model.datacollector.get_model_vars_dataframe()


def run_credit_model(params, n_steps, seed):
    """运行8.2农村信贷风险评估模型"""
    random.seed(seed)
    np.random.seed(seed)
    
    class CreditFarmer(Agent):
        def __init__(self, model, base_rate, threshold):
            super().__init__(model)
            self.wealth = random.uniform(30, 200)
            self.income = random.uniform(20, 100)
            self.credit_score = random.uniform(0.3, 1.0)
            self.debt = 0
            self.has_loan = False
            self.defaulted = False
            self.base_rate = base_rate
            self.threshold = threshold
        
        def step(self):
            # 收入波动
            if random.random() < 0.1:
                self.income *= random.uniform(0.6, 0.9)
            
            # 贷款申请
            if not self.has_loan and self.wealth < self.income * 0.5 and self.credit_score >= self.threshold:
                loan_amount = self.income * random.uniform(0.5, 1.5)
                interest_rate = self.base_rate + (1 - self.credit_score) * 0.05
                self.debt = loan_amount * (1 + interest_rate)
                self.has_loan = True
                self.wealth += loan_amount
            
            # 还款
            if self.has_loan and self.debt > 0:
                repay = min(self.income * 0.3, self.debt)
                if self.wealth >= repay:
                    self.wealth -= repay
                    self.debt -= repay
                else:
                    # 违约风险
                    if random.random() < 0.3:
                        self.defaulted = True
                        self.credit_score *= 0.7
                
                if self.debt <= 0:
                    self.has_loan = False
                    self.credit_score = min(1.0, self.credit_score * 1.05)
            
            # 正常经营收入
            self.wealth += self.income * 0.1
            self.wealth = max(5, self.wealth)
    
    class CreditModel(Model):
        def __init__(self, n_farmers, base_rate, threshold):
            super().__init__()
            for _ in range(n_farmers):
                CreditFarmer(self, base_rate, threshold)
            
            self.datacollector = DataCollector(
                model_reporters={
                    "贷款率": lambda m: sum(1 for a in m.agents if a.has_loan) / len(list(m.agents)),
                    "违约率": lambda m: sum(1 for a in m.agents if a.defaulted) / len(list(m.agents)),
                    "平均信用分": lambda m: np.mean([a.credit_score for a in m.agents]),
                    "平均财富": lambda m: np.mean([a.wealth for a in m.agents]),
                    "总债务": lambda m: sum(a.debt for a in m.agents)
                }
            )
        
        def step(self):
            self.agents.shuffle_do("step")
            self.datacollector.collect(self)
    
    model = CreditModel(params["n_farmers"], params["base_interest_rate"], params["credit_threshold"])
    
    progress = st.progress(0)
    for i in range(n_steps):
        model.step()
        progress.progress((i + 1) / n_steps)
    progress.empty()
    
    return model.datacollector.get_model_vars_dataframe()


def run_technology_diffusion_model(params, n_steps, seed):
    """运行10.2农业技术扩散模型"""
    random.seed(seed)
    np.random.seed(seed)
    
    class TechAgent(Agent):
        def __init__(self, model, is_initial_adopter):
            super().__init__(model)
            self.wealth = random.uniform(50, 200)
            self.innovation_proneness = random.random()
            self.adopted = is_initial_adopter
            self.adoption_step = 0 if is_initial_adopter else None
            self.pos = None
        
        def step(self):
            if self.adopted:
                return
            
            # 统计邻居采用情况
            neighbors = self.model.grid.get_neighbors(self.pos, moore=True)
            n_adopted = sum(1 for n in neighbors if n.adopted)
            n_total = len(list(neighbors))
            
            # 计算采用概率
            if n_total > 0:
                neighbor_effect = (n_adopted / n_total) * 0.5
            else:
                neighbor_effect = 0
            
            innovation_effect = self.innovation_proneness * 0.3
            wealth_effect = min(1.0, self.wealth / 200) * 0.2
            
            prob = neighbor_effect + innovation_effect + wealth_effect
            
            if random.random() < prob:
                self.adopted = True
                self.adoption_step = self.model.current_step
    
    class TechDiffusionModel(Model):
        def __init__(self, n_agents, grid_size, initial_ratio):
            super().__init__()
            self.grid = MultiGrid(grid_size, grid_size, torus=True)
            self.current_step = 0
            
            n_initial = int(n_agents * initial_ratio)
            
            for i in range(n_agents):
                is_initial = i < n_initial
                a = TechAgent(self, is_initial)
                x = random.randrange(grid_size)
                y = random.randrange(grid_size)
                a.pos = (x, y)
                self.grid.place_agent(a, (x, y))
            
            self.datacollector = DataCollector(
                model_reporters={
                    "采用率": lambda m: sum(1 for a in m.agents if a.adopted) / len(list(m.agents)),
                    "本期新采用": lambda m: sum(1 for a in m.agents if a.adoption_step == m.current_step),
                    "平均财富": lambda m: np.mean([a.wealth for a in m.agents])
                }
            )
        
        def step(self):
            self.current_step += 1
            self.agents.shuffle_do("step")
            self.datacollector.collect(self)
    
    model = TechDiffusionModel(params["n_agents"], params["grid_size"], params["initial_adopters"])
    
    progress = st.progress(0)
    for i in range(n_steps):
        model.step()
        progress.progress((i + 1) / n_steps)
    progress.empty()
    
    return model.datacollector.get_model_vars_dataframe()


def run_banking_contagion_model(params, n_steps, seed):
    """运行12.2-12.5银行风险传染模型 - 扩展版（张亮论文拓展）"""
    random.seed(seed)
    np.random.seed(seed)
    
    # 提取扩展参数
    enable_bailout = params.get("enable_bailout", False)
    bailout_intensity = params.get("bailout_intensity", 0.2)
    network_density = params.get("network_density", 3)
    lgd = params.get("lgd", 0.6)
    car_threshold = params.get("car_threshold", 0.08)
    stress_increment = params.get("stress_increment", 0.15)
    enable_monte_carlo = params.get("enable_monte_carlo", False)
    monte_carlo_runs = params.get("monte_carlo_runs", 1)
    
    class BankAgentExtended(Agent):
        """扩展版银行智能体 - 支持央行救助和参数化"""
        def __init__(self, model, bank_id, capital):
            super().__init__(model)
            self.bank_id = bank_id
            self.capital = capital
            self.initial_capital = capital
            self.deposits = random.uniform(capital * 5, capital * 10)
            self.loans = random.uniform(capital * 3, capital * 8)
            self.interbank_assets = 0
            self.interbank_liabilities = 0
            self.stress = 0
            self.defaulted = False
            self.received_bailout = False
            self.bailout_amount = 0
            self.contagion_source = None
        
        def step(self):
            if self.defaulted:
                return
            
            # 央行救助机制
            if enable_bailout and self.stress > 0.7 and not self.received_bailout:
                bailout = self.initial_capital * bailout_intensity
                self.capital += bailout
                self.received_bailout = True
                self.bailout_amount = bailout
                self.model.total_bailout += bailout
            
            # 资本充足率检查
            total_assets = self.loans + self.interbank_assets + self.capital
            capital_ratio = self.capital / max(1, total_assets)
            
            if capital_ratio < car_threshold:
                self.stress += stress_increment * 0.5
            
            if self.stress > 0.5 and random.random() < 0.3:
                self.defaulted = True
    
    class BankingModelExtended(Model):
        """扩展版银行体系模型 - 支持六大扩展方向"""
        def __init__(self, n_banks, network_type, shock_scenario):
            super().__init__()
            self.n_banks = n_banks
            self.network_type = network_type
            self.shock_scenario = shock_scenario
            self.current_step = 0
            self.total_bailout = 0
            
            # 创建网络 - 支持网络密度参数化
            if network_type == "scale_free":
                self.network = nx.barabasi_albert_graph(n_banks, m=network_density)
            elif network_type == "small_world":
                k = min(2 * network_density, n_banks - 1)
                self.network = nx.watts_strogatz_graph(n_banks, k=k, p=0.3)
            else:
                self.network = nx.erdos_renyi_graph(n_banks, p=0.1 + network_density * 0.02)
            
            # 创建银行
            self.banks = []
            for i in range(n_banks):
                capital = random.uniform(100, 10000)
                bank = BankAgentExtended(self, f"Bank_{i}", capital)
                self.banks.append(bank)
            
            # 计算网络中心性
            centrality = nx.degree_centrality(self.network)
            for i, bank in enumerate(self.banks):
                bank.network_centrality = centrality.get(i, 0)
            
            # 分配同业暴露
            for edge in self.network.edges():
                i, j = edge
                exposure = random.uniform(1000, 5000)
                self.banks[i].interbank_assets += exposure
                self.banks[j].interbank_liabilities += exposure
            
            # 初始冲击 - 扩展多种场景
            self._trigger_initial_shock(shock_scenario)
            
            self.datacollector = DataCollector(
                model_reporters={
                    "系统稳定性": lambda m: 1 - sum(1 for b in m.banks if b.defaulted) / m.n_banks,
                    "违约银行数": lambda m: sum(1 for b in m.banks if b.defaulted),
                    "平均压力": lambda m: np.mean([b.stress for b in m.banks]),
                    "网络聚类系数": lambda m: nx.average_clustering(m.network),
                    "救助总额": lambda m: m.total_bailout,
                    "救助银行数": lambda m: sum(1 for b in m.banks if b.received_bailout)
                }
            )
        
        def _trigger_initial_shock(self, scenario):
            """触发初始冲击 - 支持多种场景"""
            if scenario == "单个银行违约":
                # 选择中心性最高的银行
                sorted_banks = sorted(self.banks, key=lambda b: b.network_centrality, reverse=True)
                sorted_banks[0].defaulted = True
                sorted_banks[0].contagion_source = "initial_shock"
                
            elif scenario == "多银行同时违约":
                for i in range(min(3, self.n_banks)):
                    self.banks[i].defaulted = True
                    self.banks[i].contagion_source = "initial_shock"
                    
            elif scenario == "多点分散冲击":
                # 选择高、中、低中心性各一家，模拟分散冲击
                sorted_banks = sorted(self.banks, key=lambda b: b.network_centrality, reverse=True)
                n = len(sorted_banks)
                for idx in [0, n//2, n-1]:
                    sorted_banks[idx].defaulted = True
                    sorted_banks[idx].contagion_source = "multipoint_shock"
                    
            elif scenario == "渐进式冲击":
                # 不立即违约，而是存款逐步下降，在step中处理
                self.gradual_shock_remaining = 10
                
            else:  # 宏观经济冲击
                for bank in self.banks:
                    bank.stress += random.uniform(0, 0.2)
        
        def step(self):
            self.current_step += 1
            
            # 渐进式冲击处理
            if hasattr(self, 'gradual_shock_remaining') and self.gradual_shock_remaining > 0:
                for bank in self.banks:
                    bank.deposits *= 0.98  # 每步-2%
                self.gradual_shock_remaining -= 1
            
            # 传染机制 - 使用参数化LGD
            for bank in self.banks:
                if not bank.defaulted:
                    bank_idx = int(bank.bank_id.split("_")[1])
                    neighbors = list(self.network.neighbors(bank_idx))
                    for n_idx in neighbors:
                        if self.banks[n_idx].defaulted:
                            loss = bank.interbank_assets * lgd * 0.5  # 使用参数化LGD
                            bank.capital -= loss
                            bank.stress += stress_increment
                            bank.contagion_source = self.banks[n_idx].bank_id
            
            for bank in self.banks:
                bank.step()
            
            self.datacollector.collect(self)
        
        def get_network_state(self):
            """获取网络状态用于可视化"""
            return {
                'network': self.network,
                'banks': self.banks,
                'step': self.current_step
            }
    
    # 蒙特卡洛实验模式
    if enable_monte_carlo and monte_carlo_runs > 1:
        all_results = []
        mc_progress = st.progress(0)
        
        for run in range(monte_carlo_runs):
            random.seed(seed + run)
            np.random.seed(seed + run)
            
            model = BankingModelExtended(params["n_banks"], params["network_type"], params["shock_scenario"])
            for _ in range(n_steps):
                model.step()
            
            final_data = model.datacollector.get_model_vars_dataframe().iloc[-1]
            all_results.append({
                'run': run,
                '最终稳定性': final_data['系统稳定性'],
                '违约率': 1 - final_data['系统稳定性'],
                '救助总额': final_data['救助总额']
            })
            mc_progress.progress((run + 1) / monte_carlo_runs)
        
        mc_progress.empty()
        
        # 统计摘要
        results_df = pd.DataFrame(all_results)
        st.markdown("### 蒙特卡洛实验结果")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("违约率均值", f"{results_df['违约率'].mean():.2%}")
        with col2:
            st.metric("违约率标准差", f"{results_df['违约率'].std():.2%}")
        with col3:
            st.metric("95%置信区间", 
                     f"[{results_df['违约率'].quantile(0.025):.2%}, {results_df['违约率'].quantile(0.975):.2%}]")
        
        # 绘制分布图
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(results_df['违约率'], bins=15, alpha=0.7, edgecolor='black')
        ax.axvline(results_df['违约率'].mean(), color='red', linestyle='--', 
                  label=f"均值={results_df['违约率'].mean():.2%}")
        ax.set_xlabel('最终违约率')
        ax.set_ylabel('频数')
        ax.set_title(f'蒙特卡洛实验: 违约率分布 (n={monte_carlo_runs})')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()
        
        # 返回最后一次运行的详细数据
        return model.datacollector.get_model_vars_dataframe()
    
    else:
        # 单次运行模式
        model = BankingModelExtended(params["n_banks"], params["network_type"], params["shock_scenario"])
        
        progress = st.progress(0)
        for i in range(n_steps):
            model.step()
            progress.progress((i + 1) / n_steps)
        progress.empty()
        
        # 传染网络可视化
        if params.get("show_network_viz", False):
            st.markdown("### 传染网络可视化")
            fig, ax = plt.subplots(figsize=(12, 10))
            
            G = model.network
            pos = nx.spring_layout(G, seed=42)
            
            # 节点颜色
            node_colors = []
            for bank in model.banks:
                if bank.defaulted:
                    node_colors.append('red')
                elif bank.stress > 0.5:
                    node_colors.append('orange')
                elif bank.received_bailout:
                    node_colors.append('green')
                else:
                    node_colors.append('lightblue')
            
            # 节点大小
            node_sizes = [max(50, bank.initial_capital / 50) for bank in model.banks]
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax)
            nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)
            
            ax.set_title(f'银行体系传染网络 (步数={n_steps})\n红色=违约, 橙色=高压力, 绿色=已救助, 蓝色=健康',
                        fontsize=12)
            ax.axis('off')
            st.pyplot(fig)
            plt.close()
        
        return model.datacollector.get_model_vars_dataframe()


def run_flood_insurance_model(params, n_steps, seed):
    """运行洪水风险与保险ABM模型"""
    random.seed(seed)
    np.random.seed(seed)
    
    # 洪水概率配置
    flood_probs = {
        'baseline': {'low': 0.01, 'medium': 0.05, 'high': 0.10},
        '2030s': {'low': 0.015, 'medium': 0.075, 'high': 0.15},
        '2050s': {'low': 0.02, 'medium': 0.10, 'high': 0.20}
    }
    
    scenario = params["climate_scenario"]
    
    class FloodResident(Agent):
        def __init__(self, model):
            super().__init__(model)
            self.wealth = np.random.lognormal(11, 0.5)
            self.risk_perception = np.random.beta(2, 5)
            self.owns_property = random.random() < 0.65
            self.property_value = np.random.lognormal(12.5, 0.3) if self.owns_property else 0
            self.flood_zone = random.choice(['low', 'medium', 'high']) if self.owns_property else None
            self.has_insurance = False
            self.has_plpm = False
            self.flood_history = []
            self.covered_by_flood_re = random.random() < 0.85 if self.owns_property else False
        
        def step(self):
            if not self.owns_property:
                return
            
            # 保险决策
            flood_prob = flood_probs[scenario][self.flood_zone]
            expected_loss = flood_prob * self.property_value * 0.3
            
            if self.covered_by_flood_re:
                premium = 350 + 10.50
            else:
                premium = expected_loss * 1.2 + 200
            
            utility_with = -premium
            utility_without = -expected_loss * self.risk_perception
            
            if utility_with > utility_without and self.wealth > premium * 2:
                self.has_insurance = True
                self.wealth -= premium
            
            # PLPMs决策
            plpm_cost = 5000
            if not self.has_plpm and self.wealth >= plpm_cost:
                if len(self.flood_history) > 0 and random.random() < 0.34:
                    self.has_plpm = True
                    self.wealth -= plpm_cost
                elif random.random() < 0.01:
                    self.has_plpm = True
                    self.wealth -= plpm_cost
    
    class FloodModel(Model):
        def __init__(self, n_residents, climate_scenario):
            super().__init__()
            self.scenario = climate_scenario
            self.flood_re_fund = 100000
            self.all_agents = []
            
            for _ in range(n_residents):
                agent = FloodResident(self)
                self.all_agents.append(agent)
            
            self.datacollector = DataCollector(
                model_reporters={
                    "保险覆盖率": lambda m: sum(1 for a in m.all_agents if a.owns_property and a.has_insurance) / max(1, sum(1 for a in m.all_agents if a.owns_property)),
                    "PLPMs采纳率": lambda m: sum(1 for a in m.all_agents if a.owns_property and a.has_plpm) / max(1, sum(1 for a in m.all_agents if a.owns_property)),
                    "Flood_Re资金池": lambda m: m.flood_re_fund,
                    "平均财富": lambda m: np.mean([a.wealth for a in m.all_agents])
                }
            )
        
        def step(self):
            # 洪水事件
            for agent in self.all_agents:
                if agent.owns_property:
                    flood_prob = flood_probs[self.scenario][agent.flood_zone]
                    if random.random() < flood_prob:
                        flood_depth = np.random.gamma(2, 0.3)
                        agent.flood_history.append(flood_depth)
                        
                        # 损失计算
                        if flood_depth < 0.3:
                            damage_rate = 0.1
                        elif flood_depth < 0.6:
                            damage_rate = 0.3
                        elif flood_depth < 1.0:
                            damage_rate = 0.5
                        else:
                            damage_rate = 0.7
                        
                        base_damage = agent.property_value * damage_rate
                        actual_damage = base_damage * 0.25 if agent.has_plpm else base_damage
                        
                        if agent.has_insurance:
                            payout = max(0, actual_damage - 250)
                            agent.wealth -= 250
                            self.flood_re_fund -= payout * 0.8
                        else:
                            agent.wealth -= actual_damage
                        
                        agent.risk_perception = min(1.0, agent.risk_perception * 1.2)
            
            # 智能体行动
            random.shuffle(self.all_agents)
            for agent in self.all_agents:
                agent.step()
            
            # 收集数据
            self.datacollector.collect(self)
    
    model = FloodModel(params["n_residents"], params["climate_scenario"])
    
    progress = st.progress(0)
    for i in range(n_steps):
        model.step()
        progress.progress((i + 1) / n_steps)
    progress.empty()
    
    return model.datacollector.get_model_vars_dataframe()


def run_euro_crisis_model(params, n_steps, seed):
    """运行欧元区经济危机AB-SFC模型"""
    random.seed(seed)
    np.random.seed(seed)
    
    crisis = params["crisis_scenario"]
    omt = params["omt_enabled"]
    
    class Firm(Agent):
        def __init__(self, model, country):
            super().__init__(model)
            self.country = country
            self.capital = random.uniform(50, 150)
            self.production = self.capital * random.uniform(0.8, 1.2)
            self.confidence = 1.0
            self.defaulted = False
        
        def step(self):
            if self.defaulted:
                return
            
            # 生产决策
            self.production = self.capital * self.confidence * random.uniform(0.9, 1.1)
            
            # 信心恢复
            self.confidence = min(1.0, self.confidence + 0.01)
    
    class Household(Agent):
        def __init__(self, model, country):
            super().__init__(model)
            self.country = country
            self.wealth = random.uniform(20, 100)
            self.consumption = self.wealth * 0.1
            self.employed = True
        
        def step(self):
            if self.employed:
                self.wealth += random.uniform(5, 15)
            self.consumption = self.wealth * random.uniform(0.08, 0.12)
            self.wealth -= self.consumption
            self.wealth = max(5, self.wealth)
    
    class EuroCrisisModel(Model):
        def __init__(self, n_firms, n_households, crisis_scenario, omt_enabled):
            super().__init__()
            self.crisis = crisis_scenario
            self.omt = omt_enabled
            self.current_step = 0
            self.crisis_applied = False
            
            # 初始化指标
            self.policy_rate = 0.02
            self.core_spread = 0.0015
            self.periphery_spread = 0.0015
            self.core_gdp = 100
            self.periphery_gdp = 80
            self.unemployment_rate = 0.05
            self.lending_standard = 1.0
            
            # 创建智能体
            self.firms = []
            self.households = []
            
            for _ in range(n_firms):
                self.firms.append(Firm(self, 'core'))
                self.firms.append(Firm(self, 'periphery'))
            
            for _ in range(n_households):
                self.households.append(Household(self, 'core'))
                self.households.append(Household(self, 'periphery'))
            
            self.datacollector = DataCollector(
                model_reporters={
                    "核心国GDP": lambda m: m.core_gdp,
                    "外围国GDP": lambda m: m.periphery_gdp,
                    "失业率": lambda m: m.unemployment_rate,
                    "政策利率": lambda m: m.policy_rate,
                    "核心国利差": lambda m: m.core_spread * 10000,
                    "外围国利差": lambda m: m.periphery_spread * 10000,
                    "企业违约率": lambda m: sum(1 for f in m.firms if f.defaulted) / len(m.firms)
                }
            )
        
        def apply_crisis(self):
            if self.crisis == "financial_crisis":
                # 2008金融危机: 信贷紧缩+信心崩溃
                self.lending_standard = 1.5
                for firm in self.firms:
                    firm.confidence *= 0.7
                    if random.random() < 0.1:
                        firm.defaulted = True
                self.core_gdp *= 0.92
                self.periphery_gdp *= 0.88
                self.unemployment_rate = 0.12
            
            elif self.crisis == "sovereign_crisis":
                # 2010欧傺危机: 利差飙升
                self.periphery_spread = 0.05  # 500bp
                self.core_spread = 0.005
            
            elif self.crisis == "inflation_shock":
                # 2021通胀: 泰勒规则响应
                self.policy_rate = 0.04
        
        def step(self):
            self.current_step += 1
            
            # 应用危机冲击
            if self.current_step == 10 and not self.crisis_applied:
                self.apply_crisis()
                self.crisis_applied = True
            
            # OMT干预
            if self.omt and self.periphery_spread > 0.0025:
                self.periphery_spread = max(0.0025, self.periphery_spread * 0.9)
            
            # GDP恢复
            if self.crisis_applied and self.current_step > 15:
                self.core_gdp *= 1.005
                self.periphery_gdp *= 1.003
                self.unemployment_rate = max(0.05, self.unemployment_rate * 0.99)
            
            # 智能体行动
            for firm in self.firms:
                firm.step()
            for hh in self.households:
                hh.step()
            
            self.datacollector.collect(self)
    
    model = EuroCrisisModel(
        params["n_firms"], params["n_households"],
        params["crisis_scenario"], params["omt_enabled"]
    )
    
    progress = st.progress(0)
    for i in range(n_steps):
        model.step()
        progress.progress((i + 1) / n_steps)
    progress.empty()
    
    return model.datacollector.get_model_vars_dataframe()


def run_grain_market_model(params, n_steps, seed):
    """运行粮食市场政策仿真模型 (完整版)
    
    基于2024年真实数据校准，支持关税、保险、信贷、农发行收购贷款、农地流转等政策模块
    
    校准数据来源:
    - 粮食产量: 国家统计局2024年12月13日公告
    - 价格数据: 国家统计局流通领域价格监测(2024.10)
    - 保险补贴: 财政部财金〔2023〕59号
    - 贷款利率: 中国人民银行普惠金融报告(2024)
    - 农户结构: 全国人大农业法执法检查报告(2024)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    n_farmers = params["n_farmers"]
    enable_financial = params.get("enable_insurance", False) or params.get("enable_credit", False)
    
    # 尝试使用完整模型
    if GRAIN_MODEL_AVAILABLE:
        try:
            # 创建完整模型
            model = FullGrainMarketModel(
                n_farmers=n_farmers,
                random_seed=seed,
                enable_land_transfer=params.get("enable_land_transfer", False),
                enable_financial_modules=enable_financial,
                enable_policy_bank=params.get("enable_policy_bank", False)
            )
            
            # 设置关税参数
            model.government.tariff_rates[CropType.SOYBEAN] = params.get("initial_tariff_soybean", 0.01)
            model.government.tariff_rates[CropType.STAPLE_GRAIN] = 0.01
            
            # 设置生产补贴
            model.government.production_subsidies[CropType.STAPLE_GRAIN] = params.get("subsidy_staple", 100)
            model.government.production_subsidies[CropType.SOYBEAN] = params.get("subsidy_soybean", 200)
            
            # 设置保险补贴结构
            if enable_financial and model.insurance_firm:
                model.government.insurance_subsidy_structure["central"] = params.get("insurance_central", 0.45)
                model.government.insurance_subsidy_structure["province"] = params.get("insurance_province", 0.25)
                model.government.insurance_subsidy_structure["county"] = params.get("insurance_county", 0.10)
                model.government.insurance_subsidy_structure["farmer"] = params.get("insurance_farmer", 0.20)
            
            # 设置信贷参数
            if params.get("enable_credit", False) and model.rural_bank:
                model.rural_bank.base_interest_rate = params.get("base_interest_rate", 0.0413)
            
            # 设置农发行参数
            if params.get("enable_policy_bank", False) and model.policy_bank:
                model.policy_bank.base_interest_rate = params.get("policy_bank_rate", 0.0325)
            
            # 关税冲击参数
            enable_tariff_shock = params.get("enable_tariff_shock", False)
            tariff_shock_rate = params.get("tariff_shock_rate", 0.15)
            tariff_shock_step = params.get("tariff_shock_step", 5)
            
            # 运行仿真
            progress = st.progress(0)
            for i in range(n_steps):
                # 关税冲击
                if enable_tariff_shock and model.current_step == tariff_shock_step:
                    model.government.tariff_rates[CropType.SOYBEAN] = tariff_shock_rate
                
                model.step()
                progress.progress((i + 1) / n_steps)
            progress.empty()
            
            # 获取数据
            model_data = model.datacollector.get_model_vars_dataframe()
            agent_data = model.datacollector.get_agent_vars_dataframe()
            
            return model_data, agent_data, model
            
        except Exception as e:
            st.warning(f"完整模型运行失败: {e}，使用简化版本")
    
    # 简化版本（回退方案）
    tariff = params.get("initial_tariff_soybean", 0.01)
    ins_subsidy = 1 - params.get("insurance_farmer", 0.20)  # 补贴率 = 1 - 农户自缴
    
    # 校准参数
    world_price_staple = 1700    # 主粮国际价 (校准: 农业农村部2024.11)
    world_price_soybean = 2960   # 大豆国际价
    yield_staple = 6.5           # 主粮单产 (校准: 国家统计局2024)
    yield_soybean = 1.99         # 大豆单产
    cost_staple = 8000           # 主粮生产成本
    cost_soybean = 7000          # 大豆生产成本
    
    class GrainFarmer(Agent):
        """粮食市场农户主体"""
        def __init__(self, model, producer_type):
            super().__init__(model)
            self.producer_type = producer_type
            
            if producer_type == "small":
                self.land_area = random.uniform(0.3, 2.0)    # 校准: 小农户
                self.risk_aversion = random.uniform(0.5, 0.8)
            else:  # large
                self.land_area = random.uniform(5.0, 20.0)   # 校准: 规模主体
                self.risk_aversion = random.uniform(0.3, 0.5)
            
            self.wealth = self.land_area * 8000  # 校准: 农村人均收入23119元
            self.soybean_ratio = 0.3
            self.has_insurance = False
            self.income = 0
            
        def decide_crop_mix(self):
            price_staple = self.model.domestic_price_staple
            price_soybean = self.model.domestic_price_soybean
            
            profit_staple = price_staple * yield_staple - cost_staple
            profit_soybean = price_soybean * yield_soybean - cost_soybean
            
            ratio = profit_soybean / (profit_staple + profit_soybean + 0.01)
            target_ratio = ratio * (1 - 0.3 * self.risk_aversion)
            
            max_change = 0.1  # 惯性约束±10%
            if target_ratio > self.soybean_ratio + max_change:
                self.soybean_ratio += max_change
            elif target_ratio < self.soybean_ratio - max_change:
                self.soybean_ratio -= max_change
            else:
                self.soybean_ratio = target_ratio
            
            self.soybean_ratio = max(0.05, min(0.8, self.soybean_ratio))
        
        def decide_insurance(self):
            premium_rate = 0.05 * (1 - ins_subsidy)
            threshold = 0.4 - self.risk_aversion * 0.2
            self.has_insurance = random.random() < (self.risk_aversion - threshold)
        
        def produce(self):
            disaster = random.random() < 0.15  # 15%灾害概率
            loss_rate = random.uniform(0.2, 0.5) if disaster else 0
            
            area_staple = self.land_area * (1 - self.soybean_ratio)
            area_soybean = self.land_area * self.soybean_ratio
            
            output_staple = area_staple * yield_staple * (1 - loss_rate)
            output_soybean = area_soybean * yield_soybean * (1 - loss_rate)
            
            revenue = (output_staple * self.model.domestic_price_staple + 
                       output_soybean * self.model.domestic_price_soybean)
            cost = area_staple * cost_staple + area_soybean * cost_soybean
            
            if disaster and self.has_insurance:
                compensation = loss_rate * cost * 0.7  # 70%赔付
                revenue += compensation
            
            self.income = revenue - cost
            self.wealth += self.income
            self.wealth = max(1000, self.wealth)
            
            return output_staple, output_soybean
        
        def step(self):
            self.decide_crop_mix()
            self.decide_insurance()
    
    class SimpleGrainMarketModel(Model):
        def __init__(self, n_farmers, tariff_rate):
            super().__init__()
            self.tariff = tariff_rate
            self.current_step = 0
            
            self.domestic_price_staple = 2650.0  # 校准: 2024年国内价
            self.domestic_price_soybean = 4200.0
            
            # 98%小农户 + 2%规模主体 (校准: 全国人大2024)
            self.farmers = []
            n_large = int(n_farmers * 0.02)
            for i in range(n_farmers):
                if i < n_large:
                    self.farmers.append(GrainFarmer(self, "large"))
                else:
                    self.farmers.append(GrainFarmer(self, "small"))
            
            self.demand_staple = n_farmers * 50 * 0.15   # 人均主粮0.15吨/年
            self.demand_soybean = n_farmers * 50 * 0.03  # 人均大豆0.03吨/年
            
            self.datacollector = DataCollector(
                model_reporters={
                    "自给率_大豆": lambda m: m.get_self_sufficiency("soybean"),
                    "自给率_主粮": lambda m: m.get_self_sufficiency("staple"),
                    "国内价格_大豆": lambda m: m.domestic_price_soybean,
                    "国内价格_主粮": lambda m: m.domestic_price_staple,
                    "平均财富": lambda m: np.mean([f.wealth for f in m.farmers]),
                    "大豆种植比例": lambda m: np.mean([f.soybean_ratio for f in m.farmers]),
                    "参保率": lambda m: sum(1 for f in m.farmers if f.has_insurance) / len(m.farmers),
                    "平均收入": lambda m: np.mean([f.income for f in m.farmers]),
                    "违约率": lambda m: 0.0,
                    "银行总贷款": lambda m: 0.0,
                    "总保费收入": lambda m: 0.0,
                    "总赔付支出": lambda m: 0.0,
                    "财政总支出": lambda m: 0.0,
                    "关税_大豆": lambda m: m.tariff,
                    "总产量_主粮": lambda m: sum(f.land_area * (1-f.soybean_ratio) * yield_staple for f in m.farmers),
                    "总产量_大豆": lambda m: sum(f.land_area * f.soybean_ratio * yield_soybean for f in m.farmers),
                }
            )
        
        def update_price(self, total_staple, total_soybean):
            supply_staple = total_staple
            supply_soybean = total_soybean
            
            adj_staple = 1 + 0.5 * (1 - supply_staple / max(1, self.demand_staple))
            adj_soybean = 1 + 0.5 * (1 - supply_soybean / max(1, self.demand_soybean))
            
            adj_staple = np.clip(adj_staple, 0.5, 2.0)
            adj_soybean = np.clip(adj_soybean, 0.5, 2.0)
            
            self.domestic_price_staple = world_price_staple * (1 + 0.01) * adj_staple
            self.domestic_price_soybean = world_price_soybean * (1 + self.tariff) * adj_soybean
            
            self.domestic_price_staple = np.clip(self.domestic_price_staple, 1500, 5000)
            self.domestic_price_soybean = np.clip(self.domestic_price_soybean, 2500, 8000)
        
        def get_self_sufficiency(self, crop):
            if crop == "soybean":
                total = sum(f.land_area * f.soybean_ratio * yield_soybean for f in self.farmers)
                return min(1.0, total / max(1, self.demand_soybean))
            else:
                total = sum(f.land_area * (1-f.soybean_ratio) * yield_staple for f in self.farmers)
                return min(1.0, total / max(1, self.demand_staple))
        
        def step(self):
            self.current_step += 1
            for farmer in self.farmers:
                farmer.step()
            
            total_staple = 0
            total_soybean = 0
            for farmer in self.farmers:
                s, b = farmer.produce()
                total_staple += s
                total_soybean += b
            
            self.update_price(total_staple, total_soybean)
            self.datacollector.collect(self)
    
    # 运行简化模型
    model = SimpleGrainMarketModel(n_farmers, tariff)
    
    # 关税冲击参数
    enable_tariff_shock = params.get("enable_tariff_shock", False)
    tariff_shock_rate = params.get("tariff_shock_rate", 0.15)
    tariff_shock_step = params.get("tariff_shock_step", 5)
    
    progress = st.progress(0)
    for i in range(n_steps):
        if enable_tariff_shock and model.current_step == tariff_shock_step:
            model.tariff = tariff_shock_rate
        model.step()
        progress.progress((i + 1) / n_steps)
    progress.empty()
    
    model_data = model.datacollector.get_model_vars_dataframe()
    return model_data, None, model


# =============================================================================
# 主运行逻辑
# =============================================================================

if run_simulation:
    st.markdown(f"## 📈 {selected_model} - {t('simulation_results').replace('📈 ', '')}")
    
    # 根据模型选择运行对应函数
    with st.spinner(f"正在运行 {selected_model}... / Running {selected_model}..."):
        if selected_model == "6.2 农民作物选择模型":
            data = run_crop_choice_model(model_params, n_steps, random_seed)
            
            # 可视化
            col1, col2 = st.columns(2)
            with col1:
                fig1, ax1 = plt.subplots(figsize=(8, 5))
                ax1.plot(data.index, data["传统作物比例"], label="传统作物", linewidth=2)
                ax1.plot(data.index, data["新作物比例"], label="新作物", linewidth=2)
                ax1.plot(data.index, data["多样化比例"], label="多样化", linewidth=2)
                ax1.set_xlabel("时间步")
                ax1.set_ylabel("比例")
                ax1.set_title("作物选择分布变化")
                ax1.legend()
                ax1.grid(alpha=0.3)
                st.pyplot(fig1)
            
            with col2:
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                ax2.plot(data.index, data["平均财富"], color="green", linewidth=2)
                ax2.set_xlabel("时间步")
                ax2.set_ylabel("平均财富")
                ax2.set_title("农户平均财富变化")
                ax2.grid(alpha=0.3)
                st.pyplot(fig2)
        
        elif selected_model == "7.1 农民决策与环境影响模型":
            data = run_environmental_model(model_params, n_steps, random_seed)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("最终环保作物比例", f"{data['环保作物比例'].iloc[-1]:.1%}")
            with col2:
                st.metric("平均碳排放", f"{data['平均碳排放'].iloc[-1]:.2f} tCO2")
            with col3:
                st.metric("平均财富", f"{data['平均财富'].iloc[-1]:.1f}")
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].plot(data.index, data["环保作物比例"], color="green", linewidth=2)
            axes[0].set_title("环保作物采用率")
            axes[0].set_xlabel("时间步")
            axes[0].grid(alpha=0.3)
            
            axes[1].plot(data.index, data["平均碳排放"], color="red", linewidth=2)
            axes[1].set_title("平均碳排放")
            axes[1].set_xlabel("时间步")
            axes[1].grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        elif selected_model == "7.2 灌溉与水资源管理模型":
            data = run_irrigation_model(model_params, n_steps, random_seed)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("充分灌溉比例", f"{data['充分灌溉比例'].iloc[-1]:.1%}")
            with col2:
                st.metric("平均水分胁迫", f"{data['平均水分胁迫'].iloc[-1]:.2f}")
            with col3:
                st.metric("平均产量", f"{data['平均产量'].iloc[-1]:.0f} kg")
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            axes[0].stackplot(data.index, 
                            data["充分灌溉比例"], 
                            data["补充灌溉比例"],
                            data["雨养比例"],
                            labels=["充分灌溉", "补充灌溉", "雨养"],
                            colors=["#2ca02c", "#ff7f0e", "#d62728"])
            axes[0].legend(loc="upper right")
            axes[0].set_title("灌溉策略分布")
            axes[0].set_xlabel("时间步")
            
            axes[1].plot(data.index, data["平均水分胁迫"], color="brown", linewidth=2)
            axes[1].set_title("平均水分胁迫指数")
            axes[1].set_xlabel("时间步")
            axes[1].grid(alpha=0.3)
            
            axes[2].plot(data.index, data["平均产量"], color="green", linewidth=2)
            axes[2].set_title("平均作物产量")
            axes[2].set_xlabel("时间步")
            axes[2].grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        elif selected_model == "8.1 农业保险采纳模型":
            data = run_insurance_model(model_params, n_steps, random_seed)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("最终参保率", f"{data['参保率'].iloc[-1]:.1%}")
            with col2:
                st.metric("平均财富", f"{data['平均财富'].iloc[-1]:.1f}")
            with col3:
                st.metric("累计损失", f"{data['累计损失'].iloc[-1]:.0f}")
            with col4:
                st.metric("累计赔付", f"{data['累计赔付'].iloc[-1]:.0f}")
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            axes[0].plot(data.index, data["参保率"], color="blue", linewidth=2)
            axes[0].set_title("农户参保率变化")
            axes[0].set_xlabel("时间步")
            axes[0].set_ylabel("参保率")
            axes[0].grid(alpha=0.3)
            
            axes[1].plot(data.index, data["累计损失"], label="累计损失", linewidth=2)
            axes[1].plot(data.index, data["累计赔付"], label="累计赔付", linewidth=2)
            axes[1].set_title("损失与赔付")
            axes[1].set_xlabel("时间步")
            axes[1].legend()
            axes[1].grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        elif selected_model == "8.2 农村信贷风险评估模型":
            data = run_credit_model(model_params, n_steps, random_seed)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("贷款率", f"{data['贷款率'].iloc[-1]:.1%}")
            with col2:
                st.metric("违约率", f"{data['违约率'].iloc[-1]:.1%}")
            with col3:
                st.metric("平均信用分", f"{data['平均信用分'].iloc[-1]:.2f}")
            with col4:
                st.metric("总债务", f"{data['总债务'].iloc[-1]:.0f}")
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            axes[0].plot(data.index, data["贷款率"], label="贷款率", linewidth=2)
            axes[0].plot(data.index, data["违约率"], label="违约率", linewidth=2, color="red")
            axes[0].set_title("贷款与违约情况")
            axes[0].set_xlabel("时间步")
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            
            axes[1].plot(data.index, data["平均信用分"], color="green", linewidth=2)
            axes[1].set_title("平均信用评分")
            axes[1].set_xlabel("时间步")
            axes[1].grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        elif selected_model == "10.2 农业技术扩散模型":
            data = run_technology_diffusion_model(model_params, n_steps, random_seed)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("最终采用率", f"{data['采用率'].iloc[-1]:.1%}")
            with col2:
                st.metric("峰值新采用数", f"{data['本期新采用'].max():.0f}")
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # S型扩散曲线
            axes[0].plot(data.index, data["采用率"], color="green", linewidth=2)
            axes[0].set_title("技术采用率 (S型扩散曲线)")
            axes[0].set_xlabel("时间步")
            axes[0].set_ylabel("采用率")
            axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            axes[0].grid(alpha=0.3)
            
            # 新采用者分布（钟形曲线）
            axes[1].bar(data.index, data["本期新采用"], color="orange", alpha=0.7)
            axes[1].set_title("每期新采用者数量 (Rogers钟形曲线)")
            axes[1].set_xlabel("时间步")
            axes[1].set_ylabel("新采用数")
            axes[1].grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        elif selected_model == "12.2-12.5 银行风险传染模型":
            data = run_banking_contagion_model(model_params, n_steps, random_seed)
            
            # 基本指标
            st.markdown("### 核心指标")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("系统稳定性", f"{data['系统稳定性'].iloc[-1]:.1%}")
            with col2:
                st.metric("违约银行数", f"{data['违约银行数'].iloc[-1]:.0f}")
            with col3:
                st.metric("平均压力水平", f"{data['平均压力'].iloc[-1]:.2f}")
            with col4:
                st.metric("网络聚类系数", f"{data['网络聚类系数'].iloc[-1]:.3f}")
            
            # 央行救助指标（如果启用）
            if model_params.get("enable_bailout", False) and '救助总额' in data.columns:
                st.markdown("### 央行救助统计")
                col5, col6 = st.columns(2)
                with col5:
                    st.metric("救助总额", f"{data['救助总额'].iloc[-1]:,.0f}")
                with col6:
                    st.metric("救助银行数", f"{data['救助银行数'].iloc[-1]:.0f}")
            
            # 稳定性与压力变化图
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            axes[0].plot(data.index, data["系统稳定性"], color="green", linewidth=2, label="系统稳定性")
            axes[0].plot(data.index, 1 - data["系统稳定性"], color="red", linewidth=2, label="违约比例")
            axes[0].set_title("银行体系稳定性 (张亮2017理论框架)")
            axes[0].set_xlabel("时间步")
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            
            axes[1].plot(data.index, data["平均压力"], color="orange", linewidth=2)
            axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='违约阈值')
            axes[1].set_title("银行平均压力水平")
            axes[1].set_xlabel("时间步")
            axes[1].legend()
            axes[1].grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # 扩展实验配置摘要
            with st.expander("实验配置详情", expanded=False):
                st.markdown(f"""
                **基本配置**:
                - 银行数量: {model_params["n_banks"]}
                - 网络类型: {model_params["network_type"]}
                - 冲击场景: {model_params["shock_scenario"]}
                
                **扩展参数 (张亮论文拓展)**:
                - 网络密度m: {model_params.get("network_density", 3)}
                - 违约损失率LGD: {model_params.get("lgd", 0.6):.0%}
                - 资本充足率阈值: {model_params.get("car_threshold", 0.08):.0%}
                - 压力累积速率: {model_params.get("stress_increment", 0.15)}
                - 央行救助: {"启用" if model_params.get("enable_bailout", False) else "未启用"}
                - 救助强度: {model_params.get("bailout_intensity", 0.2):.0%}
                - 蒙特卡洛: {"启用" if model_params.get("enable_monte_carlo", False) else "未启用"}
                """)
            
            # 结果解读
            st.info(f"""
            **结果解读**:
            - 系统稳定性 = 1 - 违约银行比例
            - 网络聚类系数反映银行间关联紧密程度
            - 压力水平>0.5时银行有{30}%概率违约
            - 基于张亮(2017)《复杂性视角下银行体系风险传染的计算实验研究》理论框架
            """)
        
        elif selected_model == "专题: 洪水风险与保险ABM模型":
            data = run_flood_insurance_model(model_params, n_steps, random_seed)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("保险覆盖率", f"{data['保险覆盖率'].iloc[-1]:.1%}")
            with col2:
                st.metric("PLPMs采纳率", f"{data['PLPMs采纳率'].iloc[-1]:.1%}")
            with col3:
                st.metric("Flood Re资金池", f"GBP{data['Flood_Re资金池'].iloc[-1]:,.0f}")
            with col4:
                st.metric("平均财富", f"GBP{data['平均财富'].iloc[-1]:,.0f}")
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            axes[0, 0].plot(data.index, data["保险覆盖率"], color="#2E86AB", linewidth=2)
            axes[0, 0].set_title("保险覆盖率变化")
            axes[0, 0].set_xlabel("年份")
            axes[0, 0].set_ylabel("覆盖率")
            axes[0, 0].grid(alpha=0.3)
            axes[0, 0].set_ylim([0, 1])
            
            axes[0, 1].plot(data.index, data["Flood_Re资金池"], color="#A23B72", linewidth=2)
            axes[0, 1].set_title("Flood Re资金池余额")
            axes[0, 1].set_xlabel("年份")
            axes[0, 1].set_ylabel("资金(GBP)")
            axes[0, 1].axhline(y=0, color='red', linestyle=':', alpha=0.5)
            axes[0, 1].grid(alpha=0.3)
            
            axes[1, 0].plot(data.index, data["PLPMs采纳率"], color="#27AE60", linewidth=2)
            axes[1, 0].set_title("物业级防护措施(PLPMs)采纳率")
            axes[1, 0].set_xlabel("年份")
            axes[1, 0].set_ylabel("采纳率")
            axes[1, 0].grid(alpha=0.3)
            
            axes[1, 1].plot(data.index, data["平均财富"], color="#E67E22", linewidth=2)
            axes[1, 1].set_title("居民平均财富变化")
            axes[1, 1].set_xlabel("年份")
            axes[1, 1].set_ylabel("财富(GBP)")
            axes[1, 1].grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.info(f"""
            **气候情景**: {model_params["climate_scenario"]} | **居民数**: {model_params["n_residents"]}
            
            **Flood Re机制解读**:
            - 2009年前房产享受固定保费，不反映实际风险
            - 资金池<0表示政策财务不可持续
            - PLPMs可减少75%洪水损失
            """)
        
        elif selected_model == "专题: 欧元区经济危机AB-SFC模型":
            data = run_euro_crisis_model(model_params, n_steps, random_seed)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("核心国GDP", f"{data['核心国GDP'].iloc[-1]:.1f}")
            with col2:
                st.metric("外围国GDP", f"{data['外围国GDP'].iloc[-1]:.1f}")
            with col3:
                st.metric("失业率", f"{data['失业率'].iloc[-1]:.1%}")
            with col4:
                st.metric("外围国利差", f"{data['外围国利差'].iloc[-1]:.0f}bp")
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # GDP对比
            axes[0, 0].plot(data.index, data["核心国GDP"], label="核心国(德/法)", linewidth=2, color="#2E86AB")
            axes[0, 0].plot(data.index, data["外围国GDP"], label="外围国(希/葡)", linewidth=2, linestyle="--", color="#A23B72")
            axes[0, 0].set_title("GDP变化对比")
            axes[0, 0].set_xlabel("时间步")
            axes[0, 0].set_ylabel("GDP指数")
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
            axes[0, 0].axvline(x=10, color='red', linestyle=':', alpha=0.5, label='危机冲击')
            
            # 利差变化
            axes[0, 1].plot(data.index, data["核心国利差"], label="核心国利差", linewidth=2, color="#2E86AB")
            axes[0, 1].plot(data.index, data["外围国利差"], label="外围国利差", linewidth=2, linestyle="--", color="#A23B72")
            axes[0, 1].set_title("主权利差变化 (bp)")
            axes[0, 1].set_xlabel("时间步")
            axes[0, 1].set_ylabel("利差 (bp)")
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
            
            # 失业率
            axes[1, 0].plot(data.index, data["失业率"] * 100, color="#E74C3C", linewidth=2)
            axes[1, 0].set_title("失业率变化")
            axes[1, 0].set_xlabel("时间步")
            axes[1, 0].set_ylabel("失业率 (%)")
            axes[1, 0].grid(alpha=0.3)
            
            # 企业违约率
            axes[1, 1].plot(data.index, data["企业违约率"] * 100, color="#9B59B6", linewidth=2)
            axes[1, 1].set_title("企业违约率")
            axes[1, 1].set_xlabel("时间步")
            axes[1, 1].set_ylabel("违约率 (%)")
            axes[1, 1].grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            crisis_name = {
                "baseline": "基准情景",
                "financial_crisis": "2008金融危机",
                "sovereign_crisis": "2010欧傺危机",
                "inflation_shock": "2021通胀冲击"
            }
            omt_status = "启用" if model_params["omt_enabled"] else "禁用"
            
            st.info(f"""
            **危机情景**: {crisis_name[model_params["crisis_scenario"]]} | **OMT政策**: {omt_status}
            
            **核心机制**:
            - 第10期应用危机冲击，观察传导效应
            - OMT有效压缩外围国利差至25bp以内
            - 信贷紧缩通过杠杆率传导至实体经济
            """)
        
        elif selected_model == "专题: 粮食市场政策仿真模型":
            result = run_grain_market_model(model_params, n_steps, random_seed)
            if isinstance(result, tuple):
                data, agent_data, model_obj = result
            else:
                data = result
                agent_data = None
                model_obj = None
            
            # 核心指标展示
            st.markdown("### 📊 关键指标 (校准数据来源: 国家统计局/农业农村部/财政部 2024年)")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                ssr_soybean = data['自给率_大豆'].iloc[-1] if '自给率_大豆' in data.columns else data.get('大豆自给率', pd.Series([0])).iloc[-1]
                st.metric("大豆自给率", f"{ssr_soybean:.1%}", help="校准基线: 15%")
            with col2:
                ssr_staple = data['自给率_主粮'].iloc[-1] if '自给率_主粮' in data.columns else data.get('主粮自给率', pd.Series([0])).iloc[-1]
                st.metric("主粮自给率", f"{ssr_staple:.1%}", help="校准基线: 95%")
            with col3:
                ins_rate = data['参保率'].iloc[-1] if '参保率' in data.columns else data.get('保险覆盖率', pd.Series([0])).iloc[-1]
                st.metric("保险覆盖率", f"{ins_rate:.1%}", help="校准基线: 82%")
            with col4:
                avg_wealth = data['平均财富'].iloc[-1] if '平均财富' in data.columns else data.get('平均农户财富', pd.Series([0])).iloc[-1]
                st.metric("平均财富", f"{avg_wealth:,.0f}元", help="校准基线: 农村人均收入23119元")
            
            # 金融指标(如果启用)
            if model_params.get("enable_credit", False) or model_params.get("enable_insurance", False):
                col5, col6, col7, col8 = st.columns(4)
                with col5:
                    if '违约率' in data.columns:
                        st.metric("贷款违约率", f"{data['违约率'].iloc[-1]:.1%}")
                    else:
                        st.metric("贷款违约率", "--")
                with col6:
                    if '银行总贷款' in data.columns:
                        st.metric("银行总贷款", f"{data['银行总贷款'].iloc[-1]/10000:.1f}万")
                    else:
                        st.metric("银行总贷款", "--")
                with col7:
                    if '财政总支出' in data.columns:
                        st.metric("财政总支出", f"{data['财政总支出'].iloc[-1]/10000:.1f}万")
                    else:
                        st.metric("财政总支出", "--")
                with col8:
                    if '关税_大豆' in data.columns:
                        st.metric("当前大豆关税", f"{data['关税_大豆'].iloc[-1]:.0%}")
                    else:
                        st.metric("当前大豆关税", f"{model_params.get('initial_tariff_soybean', 0.01):.0%}")
            
            # 可视化
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 自给率变化
            ssr_soy_col = '自给率_大豆' if '自给率_大豆' in data.columns else '大豆自给率'
            ssr_sta_col = '自给率_主粮' if '自给率_主粮' in data.columns else '主粮自给率'
            if ssr_soy_col in data.columns:
                axes[0, 0].plot(data.index, data[ssr_soy_col] * 100, label="大豆", linewidth=2, color="#E74C3C")
            if ssr_sta_col in data.columns:
                axes[0, 0].plot(data.index, data[ssr_sta_col] * 100, label="主粮", linewidth=2, color="#27AE60")
            axes[0, 0].axhline(y=15, color='gray', linestyle='--', alpha=0.5, label='大豆校准基线(15%)')
            axes[0, 0].axhline(y=95, color='green', linestyle=':', alpha=0.5, label='主粮目标(95%)')
            axes[0, 0].set_title("粮食自给率变化 (校准: 国家粮储局)")
            axes[0, 0].set_xlabel("时间步")
            axes[0, 0].set_ylabel("自给率 (%)")
            axes[0, 0].legend(loc='best', fontsize=8)
            axes[0, 0].grid(alpha=0.3)
            
            # 价格变化
            price_soy_col = '国内价格_大豆' if '国内价格_大豆' in data.columns else '大豆国内价格'
            price_sta_col = '国内价格_主粮' if '国内价格_主粮' in data.columns else '主粮国内价格'
            if price_soy_col in data.columns:
                axes[0, 1].plot(data.index, data[price_soy_col], label="大豆", linewidth=2, color="#E74C3C")
                axes[0, 1].axhline(y=4200, color='red', linestyle=':', alpha=0.5, label='大豆校准(4200)')
            if price_sta_col in data.columns:
                axes[0, 1].plot(data.index, data[price_sta_col], label="主粮", linewidth=2, color="#27AE60")
                axes[0, 1].axhline(y=2650, color='green', linestyle=':', alpha=0.5, label='主粮校准(2650)')
            axes[0, 1].set_title("国内价格变化 (元/吨) - 校准: 国家统计局2024.10")
            axes[0, 1].set_xlabel("时间步")
            axes[0, 1].set_ylabel("价格")
            axes[0, 1].legend(loc='best', fontsize=8)
            axes[0, 1].grid(alpha=0.3)
            
            # 种植结构与保险
            plant_col = '大豆种植比例' if '大豆种植比例' in data.columns else None
            ins_col = '参保率' if '参保率' in data.columns else '保险覆盖率'
            if plant_col and plant_col in data.columns:
                axes[1, 0].plot(data.index, data[plant_col] * 100, label="大豆种植比例", linewidth=2, color="#F39C12")
            if ins_col in data.columns:
                axes[1, 0].plot(data.index, data[ins_col] * 100, label="保险覆盖率", linewidth=2, linestyle="--", color="#3498DB")
                axes[1, 0].axhline(y=82, color='blue', linestyle=':', alpha=0.5, label='校准覆盖率(82%)')
            axes[1, 0].set_title("种植结构与保险参与 - 校准: 银保监会")
            axes[1, 0].set_xlabel("时间步")
            axes[1, 0].set_ylabel("比例 (%)")
            axes[1, 0].legend(loc='best', fontsize=8)
            axes[1, 0].grid(alpha=0.3)
            
            # 农户福利
            wealth_col = '平均财富' if '平均财富' in data.columns else '平均农户财富'
            income_col = '平均收入' if '平均收入' in data.columns else None
            if wealth_col in data.columns:
                axes[1, 1].plot(data.index, data[wealth_col], label="平均财富", linewidth=2, color="#9B59B6")
                axes[1, 1].axhline(y=23119, color='purple', linestyle=':', alpha=0.5, label='农村人均收入校准')
            if income_col and income_col in data.columns:
                ax2 = axes[1, 1].twinx()
                ax2.plot(data.index, data[income_col], label="平均收入", linewidth=2, linestyle="--", color="#1ABC9C")
                ax2.set_ylabel("收入 (元)")
                ax2.legend(loc="upper right")
            axes[1, 1].set_title("农户福利变化 - 校准: 国家统计局2025.1.17")
            axes[1, 1].set_xlabel("时间步")
            axes[1, 1].set_ylabel("财富 (元)")
            axes[1, 1].legend(loc="upper left")
            axes[1, 1].grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # 政策解读
            policy_info = []
            if model_params.get("enable_tariff_shock"):
                policy_info.append(f"关税冲击: 第{model_params.get('tariff_shock_step',5)}期↑{model_params.get('tariff_shock_rate',0.15)*100:.0f}%")
            if model_params.get("enable_insurance"):
                policy_info.append(f"保险补贴: 农户自缴{model_params.get('insurance_farmer',0.20)*100:.0f}%")
            if model_params.get("enable_credit"):
                policy_info.append(f"信贷: 基准利率{model_params.get('base_interest_rate',0.0413)*100:.2f}%")
            if model_params.get("enable_policy_bank"):
                policy_info.append(f"农发行: {model_params.get('policy_bank_rate',0.0325)*100:.2f}%")
            if model_params.get("enable_land_transfer"):
                policy_info.append("农地流转: 启用")
            
            st.info(f"""
            **政策情景**: {model_params.get("policy_scenario", "自定义")} | **启用政策**: {', '.join(policy_info) if policy_info else '无'}
            
            **校准数据来源**: 
            - 粮食产量/单产: 国家统计局2024年12月13日公告
            - 价格数据: 国家统计局流通领域价格监测(2024.10)
            - 保险补贴: 财政部财金〔2023〕59号
            - 贷款利率: 人行普惠金融报告(4.13%) / 农发行(3.25%)
            - 农户结构: 全国人大农业法执法检查报告2024(小农户98%)
            
            **核心传导**: 关税↑→进口成本↑→国内价格↑→大豆收益↑→面积扩张→自给率↑
            """)
    
    # 显示数据表格
    st.markdown(f"### 📋 {t('detailed_data')}")
    with st.expander(t('view_full_data')):
        st.dataframe(data)
    
    # 数据下载
    st.markdown(f"### {t('data_export').replace('💾 ', '')}")
    csv = data.to_csv(index=True).encode('utf-8-sig')
    model_name_clean = selected_model.replace(" ", "_").replace(".", "_")
    st.download_button(
        label=t('download_data'),
        data=csv,
        file_name=f"abm_{model_name_clean}_results.csv",
        mime="text/csv"
    )

else:
    # 未运行时显示选中模型的详细原理说明
    st.info(t('run_instruction'))
    
    # 显示选中模型的详细原理
    model_info = MODEL_CATALOG[selected_model]
    
    chapter = model_info.get('chapter_zh', model_info.get('chapter', ''))
    if st.session_state.language == "English":
        chapter = model_info.get('chapter_en', chapter)
        description = model_info.get('description_en', model_info.get('description', ''))
        theory = model_info.get('theory_en', model_info.get('theory', ''))
    else:
        description = model_info.get('description_zh', model_info.get('description', ''))
        theory = model_info.get('theory_zh', model_info.get('theory', ''))
    
    st.markdown(f"## 📖 {selected_model}")
    st.markdown(f"**{chapter}** | 代码文件: `{model_info['file']}`")
    st.markdown(f"> {description}")
    
    # 显示详细原理说明
    if theory:
        st.markdown(theory)
    
    st.markdown("---")
    
    # 简要显示其他可用模型
    with st.expander(t('view_all_models')):
        model_table = []
        for name, info in MODEL_CATALOG.items():
            chapter = info.get('chapter_zh', info.get('chapter', ''))
            if st.session_state.language == "English":
                chapter = info.get('chapter_en', chapter)
                description = info.get('description_en', info.get('description', ''))
                keywords = info.get('keywords_en', info.get('keywords', []))
            else:
                description = info.get('description_zh', info.get('description', ''))
                keywords = info.get('keywords_zh', info.get('keywords', []))
            
            model_table.append({
                t('chapter'): chapter,
                t('model_name'): name,
                t('code_file'): info["file"],
                t('keywords'): ", ".join(keywords)
            })
        st.dataframe(pd.DataFrame(model_table))

# 页脚
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray; font-size: 12px;'>
    {t('footer')}
</div>
""", unsafe_allow_html=True)
