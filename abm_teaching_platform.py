"""
################################################################################
#                                                                              #
#     ABM教学实验平台 - 从原理到应用的交互式学习系统                              #
#     ABM Teaching Platform - Interactive Learning from Theory to Practice     #
#                                                                              #
################################################################################

设计思路:
    1. 左侧导航 - 学习路径(原理→方法→案例→应用)
    2. 中间内容 - 理论讲解+交互实验
    3. 右侧工具 - 参数调整+可视化
    
学习路径:
    第一章: ABM基本原理与方法论
    第二章: ABM建模标准流程
    第三章: 经典案例学习
    第四章: 金融ABM应用
    第五章: 农业金融保险ABM

作者: 肖诗顺
版本: v1.0 (教学版)
################################################################################
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from mesa import Model, Agent
from mesa.datacollection import DataCollector
import networkx as nx
from typing import Dict, List, Any, Optional
import time
import json
import os
from datetime import datetime
import urllib.parse

# ============================================================================
# 页面配置与样式
# ============================================================================

st.set_page_config(
    page_title="ABM教学实验平台",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化session state(必须在CSS之前)
if 'current_chapter' not in st.session_state:
    st.session_state.current_chapter = "第一章 ABM基本原理"
if 'current_section' not in st.session_state:
    st.session_state.current_section = "1.1 什么是基于智能体建模"
if 'progress' not in st.session_state:
    # 尝试加载保存的进度
    progress_file = "learning_progress.json"
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                st.session_state.progress = json.load(f)
        except:
            st.session_state.progress = {}
    else:
        st.session_state.progress = {}
if 'font_size' not in st.session_state:
    st.session_state.font_size = "中等"

# 字体大小映射
FONT_SIZES = {
    "小": {"base": "14px", "h1": "24px", "h2": "20px", "h3": "18px", "title": "28px"},
    "中等": {"base": "16px", "h1": "28px", "h2": "24px", "h3": "20px", "title": "32px"},
    "大": {"base": "18px", "h1": "32px", "h2": "28px", "h3": "24px", "title": "36px"},
    "特大": {"base": "20px", "h1": "36px", "h2": "32px", "h3": "28px", "title": "40px"}
}

current_font = FONT_SIZES[st.session_state.font_size]

# 自定义CSS样式 - 动态字体
st.markdown(f"""
<style>
    /* 全局字体大小 */
    html, body, [class*="css"] {{
        font-size: {current_font['base']} !important;
    }}
    
    /* 标题字体 */
    h1 {{
        font-size: {current_font['h1']} !important;
    }}
    h2 {{
        font-size: {current_font['h2']} !important;
    }}
    h3 {{
        font-size: {current_font['h3']} !important;
    }}
    
    /* 主标题 */
    .main-title {{
        font-size: {current_font['title']} !important;
        text-align: center;
        color: #1976D2;
        font-weight: bold;
        margin: 20px 0;
    }}
    
    /* 学习目标框 */
    .learning-objective {{
        background-color: #e8f4f8;
        border-left: 4px solid #2196F3;
        padding: 15px;
        margin: 15px 0;
        border-radius: 4px;
        font-size: {current_font['base']};
    }}
    
    /* 理论讲解框 */
    .theory-box {{
        background-color: #fff3e0;
        border-left: 4px solid #FF9800;
        padding: 15px;
        margin: 15px 0;
        border-radius: 4px;
        font-size: {current_font['base']};
    }}
    
    /* 实验操作框 */
    .practice-box {{
        background-color: #f1f8e9;
        border-left: 4px solid #4CAF50;
        padding: 15px;
        margin: 15px 0;
        border-radius: 4px;
        font-size: {current_font['base']};
    }}
    
    /* 关键概念 */
    .key-concept {{
        background-color: #fce4ec;
        border-left: 4px solid #E91E63;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 4px;
        font-weight: 500;
        font-size: {current_font['base']};
    }}
    
    /* 实验卡片 */
    .experiment-card {{
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    /* 章节标题 */
    .chapter-title {{
        color: #1976D2;
        font-size: {current_font['h1']};
        font-weight: bold;
        margin: 20px 0 10px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #2196F3;
    }}
    
    /* 按钮字体 */
    .stButton button {{
        font-size: {current_font['base']} !important;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 教学内容数据结构
# ============================================================================

LEARNING_PATH = {
    "第一章 ABM基本原理": {
        "sections": [
            "1.1 什么是基于智能体建模",
            "1.2 ABM的核心概念",
            "1.3 为什么需要ABM",
            "1.4 ABM与传统建模方法的区别"
        ],
        "icon": "📚"
    },
    "第二章 ABM建模流程": {
        "sections": [
            "2.1 问题定义与边界设定",
            "2.2 主体设计与行为规则",
            "2.3 环境设计与交互机制",
            "2.4 模型验证与校准"
        ],
        "icon": "🔧"
    },
    "第三章 经典案例实验": {
        "sections": [
            "3.1 森林火灾传播模型",
            "3.2 病毒传播模型",
            "3.3 Schelling居住隔离模型",
            "3.4 技术扩散模型"
        ],
        "icon": "🧪"
    },
    "第四章 金融ABM应用": {
        "sections": [
            "4.1 银行风险传染模型",
            "4.2 信贷市场定价模型",
            "4.3 激励机制设计模型"
        ],
        "icon": "💰"
    },
    "第五章 农业金融保险ABM": {
        "sections": [
            "5.1 农户作物选择模型",
            "5.2 农业保险采纳模型",
            "5.3 农村信贷风险模型",
            "5.4 粮食市场综合模型"
        ],
        "icon": "🌾"
    }
}

# ============================================================================
# 理论内容库
# ============================================================================

THEORY_CONTENT = {
    "1.1 什么是基于智能体建模": {
        "learning_objective": """
        **学习目标:**
        - 理解ABM的定义与本质
        - 掌握智能体(Agent)的基本特征
        - 了解ABM在社会科学中的应用价值
        """,
        "theory": """
        ### 定义
        
        **基于智能体建模(Agent-Based Modeling, ABM)**是一种自底向上的计算建模方法,
        通过模拟独立决策的智能体及其交互,研究复杂系统的涌现行为。
        
        ### 智能体的核心特征
        
        1. **自主性(Autonomy)**: 智能体能够独立做出决策,无需中央控制
        2. **异质性(Heterogeneity)**: 不同智能体可以具有不同的属性和行为规则
        3. **局部交互(Local Interaction)**: 智能体只能感知和影响其局部环境
        4. **适应性(Adaptability)**: 智能体可以根据经验调整行为策略
        
        ### 涌现性(Emergence)
        
        系统层面的宏观模式不是预先设定的,而是从微观主体的交互中**自发涌现**出来的。
        这是ABM最重要的特征之一。
        
        **经典例子:**
        - 鸟群的集体飞行模式
        - 交通拥堵的形成
        - 金融市场的价格波动
        - 疾病的传播路径
        """,
        "key_concepts": [
            "智能体(Agent): 具有自主决策能力的基本单元",
            "环境(Environment): 智能体活动的空间和条件",
            "交互(Interaction): 智能体之间以及智能体与环境之间的相互作用",
            "涌现(Emergence): 宏观模式从微观交互中自发产生"
        ]
    },
    
    "1.2 ABM的核心概念": {
        "learning_objective": """
        **学习目标:**
        - 掌握ABM的三大核心要素
        - 理解智能体的决策机制
        - 学习如何设计交互规则
        """,
        "theory": """
        ### ABM的三大核心要素
        
        #### 1. 智能体(Agents)
        
        智能体是模型的基本构成单元,具有以下特征:
        
        - **状态变量**: 描述智能体的属性(如财富、年龄、风险偏好)
        - **行为规则**: 定义智能体如何感知环境和做出决策
        - **学习能力**: 智能体如何从经验中更新策略
        
        **示例 - 农户智能体:**
        ```python
        class FarmerAgent:
            状态变量:
                - land_area: 土地面积
                - wealth: 财富水平
                - risk_aversion: 风险厌恶程度
            
            行为规则:
                - decide_crop_mix(): 选择种植作物
                - decide_insurance(): 是否购买保险
                - learn_from_neighbors(): 从邻居学习
        ```
        
        #### 2. 环境(Environment)
        
        环境为智能体提供活动空间和资源:
        
        - **空间结构**: 网格、网络、连续空间
        - **资源分布**: 如土地肥力、水资源
        - **外部冲击**: 如自然灾害、政策变化
        
        #### 3. 交互规则(Interaction Rules)
        
        定义智能体如何相互影响:
        
        - **直接交互**: 交易、信息交换、模仿
        - **间接交互**: 通过价格机制、公共资源竞争
        - **网络效应**: 基于社会网络的影响传播
        
        ### 决策机制示例
        
        **期望效用理论**:
        农户在有保险(I)和无保险(NI)之间选择:
        
        ```
        EU(保险) = (1-p)·U(W-Premium) + p·U(W-Premium-Loss+Payout)
        EU(无保险) = (1-p)·U(W) + p·U(W-Loss)
        
        如果 EU(保险) > EU(无保险), 则购买保险
        ```
        
        其中:
        - p: 灾害发生概率
        - W: 初始财富
        - U(): 效用函数(通常为凹函数,体现风险厌恶)
        """,
        "key_concepts": [
            "状态变量: 描述智能体当前状态的属性集合",
            "行为规则: 从感知到决策的映射函数",
            "局部感知: 智能体只能观察其邻域内的信息",
            "异质性建模: 允许不同类型的智能体共存"
        ]
    },
    
    "1.3 为什么需要ABM": {
        "learning_objective": """
        **学习目标:**
        - 理解ABM在复杂系统研究中的独特优势
        - 认识传统建模方法的局限性
        - 掌握ABM适用的典型场景
        """,
        "theory": """
        ### ABM的独特优势
        
        #### 1. 捕捉异质性(Heterogeneity)
        
        **传统方法**: 假设"代表性个体"(Representative Agent)
        - 所有人具有相同的偏好和行为
        - 忽略个体差异对系统的影响
        
        **ABM方法**: 显式建模个体差异
        - 不同风险偏好、财富水平、信息获取能力
        - 异质性本身成为研究对象
        
        **实例**: 金融危机中,大银行和小银行的行为差异对系统稳定性的影响无法用"代表性银行"模型刻画。
        
        #### 2. 刻画涌现性(Emergence)
        
        **涌现**: 系统层面的模式无法简单归因于个体属性,而是从交互中"涌现"出来
        
        **经典案例**:
        - **交通拥堵**: 没有单个司机"想"造成拥堵,但集体决策导致拥堵涌现
        - **金融泡沫**: 理性个体的投资决策可能导致非理性的市场泡沫
        - **技术锁定**: 局部最优选择导致全局次优技术路径
        
        #### 3. 模拟非线性动态(Non-linear Dynamics)
        
        传统线性模型难以处理:
        - **反馈循环**: 价格上涨→预期改变→需求增加→价格进一步上涨
        - **阈值效应**: 当某变量超过临界点,系统行为突变
        - **路径依赖**: 历史决策约束未来选择空间
        
        ABM天然适合建模这些特征。
        
        #### 4. 政策仿真"实验室"(Policy Laboratory)
        
        **实验经济学的困境**: 
        - 伦理约束: 不能用真实经济做危险实验
        - 成本高昂: 大规模政策试验代价巨大
        - 不可重复: 无法控制所有变量
        
        **ABM的优势**:
        - 可以"穿越时空"做反事实分析(Counterfactual)
        - 零成本测试多种政策组合
        - 完全可控的"对照实验"
        
        ### ABM适用的典型场景
        
        | 场景类型 | 特征 | 应用实例 |
        |---------|------|----------|
        | **网络传播** | 个体通过网络相互影响 | 疾病传播、谣言扩散、技术采纳 |
        | **市场机制** | 买卖双方分散决策、价格涌现 | 拍卖市场、金融市场、劳动力市场 |
        | **空间过程** | 地理位置影响交互 | 城市扩张、居住隔离、土地利用 |
        | **学习适应** | 主体根据经验调整策略 | 组织学习、演化博弈、创新扩散 |
        | **政策评估** | 需考虑异质性和间接效应 | 税收政策、补贴方案、监管措施 |
        
        ### 何时不应使用ABM?
        
        - 问题可用简单数学模型充分刻画
        - 个体异质性和交互不重要
        - 缺乏微观数据做参数校准
        - 需要解析解而非数值模拟
        """,
        "key_concepts": [
            "代表性个体谬误: 忽略异质性可能导致错误结论",
            "涌现性: 宏观模式不是微观属性的简单加总",
            "反事实分析: ABM允许探索'如果...会怎样'的情景",
            "政策实验室: 低成本、可重复的政策测试平台"
        ]
    },
    
    "1.4 ABM与传统建模方法的区别": {
        "learning_objective": """
        **学习目标:**
        - 对比ABM与方程模型的差异
        - 理解自底向上与自顶向下的建模思路
        - 认识不同方法的适用边界
        """,
        "theory": """
        ### 建模范式的根本区别
        
        #### 1. 自顶向下 vs 自底向上
        
        **传统方程模型(Top-Down)**:
        ```
        宏观层面 → 直接建模聚合变量关系
        例: GDP = C + I + G + (X-M)
        ```
        
        **ABM方法(Bottom-Up)**:
        ```
        微观层面 → 个体决策与交互 → 宏观模式涌现
        例: 消费者决策 + 企业决策 → 市场均衡价格涌现
        ```
        
        #### 2. 方法对比表
        
        | 维度 | 传统方程模型 | 基于智能体模型(ABM) |
        |------|-------------|--------------------|
        | **建模单元** | 聚合变量(如总产出) | 异质性个体 |
        | **求解方式** | 解析解/数值求解方程 | 计算机仿真 |
        | **均衡概念** | 假设均衡存在 | 均衡可能涌现(也可能不均衡) |
        | **理性假设** | 完全理性/有限理性 | 可建模任意决策机制 |
        | **异质性处理** | 困难(维度诅咒) | 自然支持 |
        | **网络效应** | 通常忽略 | 核心要素 |
        | **政策评估** | 参数扰动 | 行为仿真 |
        | **可解释性** | 数学推导清晰 | 需通过仿真理解 |
        
        #### 3. 典型案例对比
        
        **问题**: 研究最低工资政策对就业的影响
        
        **传统劳动经济学模型**:
        ```
        劳动供给: Ls = f(W)  (上升)
        劳动需求: Ld = g(W)  (下降)
        
        均衡: Ls = Ld 决定均衡工资W*
        
        政策冲击: 最低工资Wmin > W* 
        → 需求下降 → 失业增加
        ```
        
        **ABM模型**:
        ```python
        智能体:
        - 异质性企业(大/中/小,不同利润率)
        - 异质性工人(不同技能水平)
        
        行为规则:
        - 企业: 根据利润决定雇佣/解雇
        - 工人: 搜寻工作,接受/拒绝offer
        
        涌现结果:
        - 部分低利润企业退出
        - 高技能工人可能受益
        - 低技能工人失业风险上升
        - 整体就业效应取决于企业结构
        ```
        
        ABM可以捕捉:
        - 企业异质性的作用
        - 工人搜寻行为
        - 动态调整过程
        - 不同群体的差异化影响
        
        #### 4. 互补而非替代
        
        **最佳实践**: 将ABM与传统方法结合
        
        1. **理论→ABM**: 用ABM检验理论的微观基础
           - 例: Stiglitz-Weiss信贷配给理论 → ABM验证动态过程
        
        2. **ABM→理论**: 从仿真发现新规律,再提炼理论
           - 例: Schelling隔离模型 → 弱偏好导致强隔离的理论洞见
        
        3. **方程+ABM**: 宏观用方程,微观用ABM
           - 例: 宏观经济DSGE框架 + 微观家庭ABM
        
        ### 选择建模方法的决策树
        
        ```
        问题是否涉及异质性个体交互?
        ├─ 否 → 考虑传统方程模型
        └─ 是 → 个体数量是否可枚举?
            ├─ 少(<10) → 博弈论模型
            └─ 多(>10) → 涌现性是否重要?
                ├─ 否 → 代表性个体模型可能足够
                └─ 是 → **使用ABM**
        ```
        """,
        "key_concepts": [
            "自底向上建模: 从微观个体行为构建宏观模式",
            "计算模型: 通过仿真而非方程求解获得结果",
            "方法互补: ABM与传统模型各有优势,应结合使用",
            "维度诅咒: 传统方法难以处理大量异质性个体"
        ]
    },
    
    # 第二章
    "2.1 问题定义与边界设定": {
        "learning_objective": """
        **学习目标:**
        - 掌握清晰定义研究问题的方法
        - 学会设定模型边界和约束
        - 理解抽象简化的原则
        """,
        "theory": """
        ### 模型开发的第一步:  明确研究问题
        
        #### 1. 研究问题的三个维度
        
        **What - 研究什么现象?**
        - 明确要解释或预测的宏观模式
        - 示例: “金融危机中风险如何在银行间传染?”
        
        **Why - 为什么需要ABM?**
        - 传统方法的局限性
        - ABM能提供的独特洞见
        - 示例: “需要考虑银行网络结构的异质性”
        
        **How - 如何衡量结果?**
        - 定义关键输出指标
        - 示例: “使用系统性风险指数、违约银行比例”
        
        #### 2. 设定模型边界
        
        **空间边界**:
        - 地理范围: 国家/地区/全球?
        - 空间粒度: 点/网格/连续空间?
        - 示例: “仅考虑中国银行间市场,忽略国际跨境传染”
        
        **时间边界**:
        - 仿真长度: 短期(天)/中期(月)/长期(年)?
        - 时间步长: 一个时间步代表多长实际时间?
        - 示例: “仿真100个时间步,每步=1个交易日”
        
        **主体边界**:
        - 包含哪些类型的主体?
        - 不包含哪些主体?
        - 示例: “包含银行,但忽略中央银行和监管机构”
        
        #### 3. 抽象简化的KISS原则
        
        **KISS = Keep It Simple, Stupid**
        
        ```
        简单模型优势:
        ✓ 更容易理解和交流
        ✓ 更容易调试和验证
        ✓ 更容易识别关键机制
        ✓ 计算效率更高
        
        复杂模型风险:
        ✗ 过多参数难以校准
        ✗ 难以判断哪个因素驱动结果
        ✗ 增加并不一定提高预测能力
        ```
        
        **渐进式复杂化策略**:
        ```
        v1.0: 最简化版本(核心机制)
         ↓ 验证 → 如果不足
        v1.1: 添加一个关键要素
         ↓ 验证 → 如果不足
        v1.2: 继续扩展...
        ```
        
        #### 4. 实例: 农户作物选择模型
        
        **问题定义**:
        > 研究价格波动如何影响农户的作物种植结构
        
        **模型边界**:
        - **空间**: 单一区域,不考虑地理差异(简化)
        - **时间**: 20个生产季,每季=1步
        - **主体**: 仅包含农户,不包含收购商、银行(简化)
        - **作物**: 只考虑两种作物(玉米vs大豆)
        
        **关键输出**:
        - 各作物种植面积比例的时间演化
        - 农户平均收益的波动性
        
        #### 5. 常见错误
        
        | 错误 | 例子 | 正确做法 |
        |------|------|----------|
        | **边界过宽** | “模型包含所有因素” | 聚焦核心问题,逐步扩展 |
        | **缺乏量化指标** | “观察系统行为” | 明确定义可测量的输出变量 |
        | **目标不清** | “建一个真实的模型” | 先回答具体研究问题 |
        | **过度抽象** | 忽略关键异质性 | 保留对问题重要的差异 |
        """,
        "key_concepts": [
            "KISS原则: 从简单模型开始,渐进式复杂化",
            "边界设定: 明确空间、时间和主体范围",
            "关键输出: 定义可测量的宏观指标",
            "问题驱动: 以研究问题指导模型设计"
        ]
    },
    
    "2.2 主体设计与行为规则": {
        "learning_objective": """
        **学习目标:**
        - 掌握智能体设计的ODD协议
        - 学会定义状态变量和行为规则
        - 理解决策机制的建模方法
        """,
        "theory": """
        ### ODD协议: ABM模型描述标准
        
        **ODD = Overview, Design concepts, Details**
        
        #### 1. Overview(概述) - 三个关键问题
        
        **1.1 Purpose(目的)**
        - 模型要回答什么问题?
        - 面向哪些读者/用户?
        
        **1.2 Entities(实体)**
        - 模型中有哪些类型的智能体?
        - 每种智能体有哪些状态变量?
        
        **1.3 Process(过程)**
        - 智能体如何决策和互动?
        - 模型如何按时间步进?
        
        #### 2. 智能体设计模板
        
        ```python
        class AgentTemplate:
            # 1. 状态变量 (State Variables)
            unique_id: int          # 唯一标识
            position: (x, y)        # 空间位置
            wealth: float           # 财务状态
            type: str               # 类型分类
            
            # 2. 感知能力 (Perception)
            def perceive_environment():
                # 观察周围环境
                # 获取邻居信息
                pass
            
            # 3. 决策机制 (Decision)
            def make_decision():
                # 基于感知做出选择
                # 可能包含随机性
                pass
            
            # 4. 行动执行 (Action)
            def take_action():
                # 执行决策结果
                # 更新自身状态
                pass
            
            # 5. 学习更新 (Learning)
            def update_strategy():
                # 根据经验调整行为
                pass
        ```
        
        #### 3. 决策机制的三种建模方式
        
        **方式1: 理性决策(Rational Choice)**
        - 最优化目标函数
        - 适用: 经济主体、机构决策
        
        ```python
        def decide_crop(self):
            expected_profit = {}
            for crop in ['玉米', '大豆']:
                profit = self.calc_expected_profit(crop)
                expected_profit[crop] = profit
            # 选择期望利润最高的作物
            return max(expected_profit, key=expected_profit.get)
        ```
        
        **方式2: 启发式规则(Heuristic Rules)**
        - “如果...  则...” 规则
        - 适用: 简单决策、本能反应
        
        ```python
        def decide_insurance(self):
            # 启发式规则
            if self.wealth < 5000:
                return False  # 太穷买不起
            elif self.risk_aversion > 0.7:
                return True   # 风险厌恶者一定买
            else:
                # 看邻居怎么做
                neighbor_insured_rate = self.get_neighbor_insurance_rate()
                return random.random() < neighbor_insured_rate
        ```
        
        **方式3: 学习适应(Learning & Adaptation)**
        - 根据历史经验调整策略
        - 适用: 长期演化、策略探索
        
        ```python
        def update_crop_preference(self):
            # 强化学习: 增加成功策略的权重
            last_profit = self.history[-1]['profit']
            last_crop = self.history[-1]['crop']
            
            if last_profit > self.avg_profit:
                self.crop_preference[last_crop] += 0.1  # 增加偏好
            else:
                self.crop_preference[last_crop] -= 0.05 # 降低偏好
        ```
        
        #### 4. 实例: 农户Agent详细设计
        
        ```python
        class FarmerAgent:
            # 状态变量
            land_area: float         # 土地面积(公顷)
            wealth: float            # 财富(元)
            risk_aversion: float     # 风险厌恶[0,1]
            age: int                 # 年龄
            education: str           # 教育水平
            neighbors: List[Agent]   # 邻居列表
            
            # 决策变量
            crop_choice: str         # 本季作物选择
            has_insurance: bool      # 是否投保
            loan_amount: float       # 贷款金额
            
            # 行为规则
            def step(self):
                # 1. 感知: 获取价格信息
                corn_price = self.model.corn_price
                soybean_price = self.model.soybean_price
                
                # 2. 决策: 选择作物
                self.crop_choice = self.decide_crop(
                    corn_price, soybean_price
                )
                
                # 3. 决策: 是否贷款
                if self.wealth < self.production_cost:
                    self.loan_amount = self.apply_for_loan()
                
                # 4. 决策: 是否投保
                self.has_insurance = self.decide_insurance()
                
                # 5. 生产: 实现产量
                yield_per_area = self.calc_yield()
                self.total_yield = self.land_area * yield_per_area
                
                # 6. 销售: 获得收入
                self.income = self.total_yield * self.get_sell_price()
                
                # 7. 更新: 财富变化
                self.wealth += self.income - self.cost
                
                # 8. 学习: 调整策略
                self.update_preferences()
        ```
        
        #### 5. 常见问题
        
        **Q1: 智能体该有多详细?**
        - 只包含与研究问题直接相关的属性
        - 避免“过度真实”的误区
        
        **Q2: 如何处理随机性?**
        - 在关键决策点引入随机扰动
        - 使用固定随机种子以便重现结果
        
        **Q3: 是否需要学习机制?**
        - 短期模型(<20步)可以省略
        - 长期模型应考虑策略调整
        """,
        "key_concepts": [
            "ODD协议: 模型描述的国际标准",
            "感知-决策-行动: 智能体的基本循环",
            "启发式规则: 简单但有效的决策方式",
            "学习适应: 智能体根据经验调整策略"
        ]
    },
    
    "2.3 环境设计与交互机制": {
        "learning_objective": """
        **学习目标:**
        - 理解环境在ABM中的作用
        - 掌握不同空间结构的选择
        - 学会设计主体间交互规则
        """,
        "theory": """
        ### 环境的三大功能
        
        #### 1. 作为智能体的活动空间
        
        **空间结构选择**:
        
        | 类型 | 特点 | 适用场景 | Mesa实现 |
        |------|------|----------|----------|
        | **网格(Grid)** | 离散单元,四/八邻居 | 土地利用,城市模拟 | `MultiGrid`, `SingleGrid` |
        | **网络(Network)** | 节点+连边,任意拓扑 | 社会网络,金融传染 | `NetworkGrid` |
        | **连续空间(Continuous)** | 实数坐标(x,y) | 生物移动,物理仿真 | `ContinuousSpace` |
        | **无空间** | 不考虑地理位置 | 金融市场,抽象交互 | 不使用Space类 |
        
        **示例: 网格空间设计**
        ```python
        from mesa.space import MultiGrid
        
        class MyModel(Model):
            def __init__(self, width=50, height=50):
                # 创建50x50网格,允许多个智能体占据同一格子
                self.grid = MultiGrid(width, height, torus=True)
                # torus=True: 环面边界(左右/上下相连)
        ```
        
        #### 2. 作为资源分布载体
        
        **环境属性**:
        - 土壤肥力分布
        - 水资源可用性
        - 污染浓度
        
        **示例: 土壤质量地图**
        ```python
        class LandModel(Model):
            def __init__(self, width, height):
                self.grid = MultiGrid(width, height, torus=False)
                # 为每个网格赋予土壤质量属性
                self.soil_quality = np.random.uniform(0.5, 1.0, (width, height))
            
            def get_soil_quality(self, x, y):
                return self.soil_quality[x][y]
        ```
        
        #### 3. 作为动态过程发生器
        
        **外部冲击**:
        - 自然灾害(干旱、洪水、虫灾)
        - 价格波动
        - 政策变化
        
        **示例: 灾害事件生成**
        ```python
        class DisasterModel(Model):
            def step(self):
                # 每期10%概率发生干旱
                if random.random() < 0.1:
                    # 随机选择受灾区域
                    disaster_zone = self.select_disaster_zone()
                    # 降低该区域产量30-70%
                    loss_rate = random.uniform(0.3, 0.7)
                    self.apply_disaster(disaster_zone, loss_rate)
        ```
        
        ### 交互机制设计
        
        #### 1. 直接交互 - 智能体之间
        
        **类型1: 邻居交互**
        ```python
        def interact_with_neighbors(self):
            # 获取摩尔邻域(八邻居)的所有智能体
            neighbors = self.model.grid.get_neighbors(
                self.pos,
                moore=True,   # 摩尔邻域(八邻居)
                include_center=False,
                radius=1       # 邻域半径
            )
            
            # 介居学习: 观察邻居的作物选择
            neighbor_crops = [n.crop_choice for n in neighbors]
            most_common = max(set(neighbor_crops), key=neighbor_crops.count)
            
            # 如果邻居大多数都种某作物,我也跟风
            if neighbor_crops.count(most_common) > len(neighbors) * 0.6:
                self.crop_choice = most_common
        ```
        
        **类型2: 市场交易**
        ```python
        def market_trade(self, buyer, seller):
            # 买家出价
            bid_price = buyer.calc_bid_price()
            # 卖家要价
            ask_price = seller.calc_ask_price()
            
            # 价格合适则成交
            if bid_price >= ask_price:
                trade_price = (bid_price + ask_price) / 2
                quantity = min(buyer.demand, seller.supply)
                # 执行交易
                buyer.wealth -= trade_price * quantity
                seller.wealth += trade_price * quantity
                return True
            return False
        ```
        
        #### 2. 间接交互 - 通过环境
        
        **公共资源竞争**
        ```python
        class WaterResourceModel(Model):
            def __init__(self):
                self.total_water = 10000  # 总水资源
            
            def step(self):
                # 1. 所有农户提交用水需求
                demands = [farmer.water_demand for farmer in self.farmers]
                total_demand = sum(demands)
                
                # 2. 水资源不足时按比例分配
                if total_demand > self.total_water:
                    ratio = self.total_water / total_demand
                    for farmer in self.farmers:
                        farmer.water_allocated = farmer.water_demand * ratio
                else:
                    for farmer in self.farmers:
                        farmer.water_allocated = farmer.water_demand
        ```
        
        **价格信号传递**
        ```python
        class PriceSignalModel(Model):
            def update_price(self):
                # 根据总供给和总需求调整价格
                total_supply = sum([f.output for f in self.farmers])
                total_demand = self.market.demand
                
                # 简单价格调整规则
                if total_supply < total_demand:
                    self.price *= 1.1  # 供不应求→价格上涨
                elif total_supply > total_demand * 1.2:
                    self.price *= 0.9  # 供过于求→价格下降
                
                # 所有农户观察到新价格
                for farmer in self.farmers:
                    farmer.observe_price(self.price)
        ```
        
        #### 3. 网络交互
        
        **社会网络上的信息传播**
        ```python
        import networkx as nx
        from mesa.space import NetworkGrid
        
        class SocialNetworkModel(Model):
            def __init__(self, n_agents=100):
                # 创建小世界网络
                G = nx.watts_strogatz_graph(n_agents, k=6, p=0.1)
                self.grid = NetworkGrid(G)
                
                # 将智能体放置到网络节点
                for i, node in enumerate(G.nodes()):
                    agent = FarmerAgent(i, self)
                    self.grid.place_agent(agent, node)
            
            def spread_information(self, agent):
                # 获取网络邻居
                neighbors = self.grid.get_neighbors(agent.pos)
                
                # 向邻居传播信息(如保险采纳)
                for neighbor in neighbors:
                    if agent.has_insurance and not neighbor.has_insurance:
                        # 有一定概率被说服
                        if random.random() < 0.2:
                            neighbor.consider_insurance()
        ```
        
        ### 实例: Schelling隔离模型
        
        **问题**: 为什么个体的微弱偏好会导致宏观的强隔离?
        
        ```python
        class SchellingAgent(Agent):
            def __init__(self, unique_id, model, agent_type):
                super().__init__(unique_id, model)
                self.type = agent_type  # 0或1(两个族群)
                self.happy = False
            
            def step(self):
                # 1. 感知: 观察邻居
                neighbors = self.model.grid.get_neighbors(
                    self.pos, moore=True, include_center=False
                )
                
                # 2. 计算同类邻居比例
                similar = sum(1 for n in neighbors if n.type == self.type)
                similarity_ratio = similar / len(neighbors) if neighbors else 0
                
                # 3. 决策: 是否满意(偏好阈值=30%)
                self.happy = similarity_ratio >= 0.3
                
                # 4. 行动: 不满意则移动
                if not self.happy:
                    self.model.grid.move_to_empty(self)
        ```
        
        **涌现结果**: 即使每个人只需西30%同类邻居,最终会形成高度隔离的社区。
        """,
        "key_concepts": [
            "空间结构: 选择网格/网络/连续空间根据问题特点",
            "直接交互: 智能体之间的主动沟通与影响",
            "间接交互: 通过共享资源或价格信号的联系",
            "网络拓扑: 社会网络结构影响信息扩散速度"
        ]
    },
    
    "2.4 模型验证与校准": {
        "learning_objective": """
        **学习目标:**
        - 理解验证与校准的区别
        - 掌握验证的多种方法
        - 学会进行敏感性分析
        """,
        "theory": """
        ### 验证 vs 校准
        
        | 概念 | 目的 | 方法 | 时机 |
        |------|------|------|------|
        | **验证(Verification)** | 确保模型按设计工作 | 代码审查,单元测试 | 开发过程中 |
        | **校准(Calibration)** | 调整参数匹配现实 | 参数估计,数据拟合 | 模型完成后 |
        | **验证(Validation)** | 评估模型预测能力 | 与实际数据对比 | 校准后 |
        
        #### 1. 验证(Verification) - 模型是否正确
        
        **方法1: 极端值测试**
        ```python
        # 测试: 当所有农户风险厌恶=1时,应全部投保
        def test_risk_averse_farmers():
            model = InsuranceModel(n_farmers=100)
            for farmer in model.farmers:
                farmer.risk_aversion = 1.0  # 极端风险厌恶
            
            model.step()
            insurance_rate = sum([f.has_insurance for f in model.farmers]) / 100
            assert insurance_rate > 0.95, "高风险厌恶者应全部投保"
        ```
        
        **方法2: 单调性检验**
        ```python
        # 测试: 保费提高→投保率下降(应单调递减)
        def test_price_sensitivity():
            results = []
            for premium in [100, 200, 300, 400, 500]:
                model = InsuranceModel(premium_rate=premium)
                model.run(steps=10)
                insurance_rate = model.get_insurance_rate()
                results.append((premium, insurance_rate))
            
            # 检验是否递减
            for i in range(len(results)-1):
                assert results[i][1] >= results[i+1][1], "保费↑应导致投保率↓"
        ```
        
        **方法3: 质量保证(Trace)**
        ```python
        # 记录关键变量的每步变化
        class FarmerAgent:
            def step(self):
                print(f"t={self.model.schedule.time}, Agent {self.unique_id}:")
                print(f"  Wealth: {self.wealth}")
                print(f"  Decision: {self.crop_choice}")
                # ... 执行决策
        ```
        
        #### 2. 校准(Calibration) - 参数调整
        
        **方法比较**:
        
        | 方法 | 原理 | 优点 | 缺点 |
        |------|------|------|------|
        | **手工调整** | 反复试错 | 简单直观 | 耗时,不系统 |
        | **网格搜索** | 遍历参数组合 | 全面 | 维度诅咒 |
        | **随机优化** | 进化算法/贝叶斯优化 | 高效 | 需要专业工具 |
        | **机器学习** | 代理模型+参数推断 | 智能 | 复杂度高 |
        
        **示例: 网格搜索校准**
        ```python
        import itertools
        
        def calibrate_model():
            # 定义参数空间
            risk_aversion_values = [0.3, 0.5, 0.7]
            price_sensitivity_values = [0.1, 0.2, 0.3]
            
            # 实际观测数据
            observed_insurance_rate = 0.45
            
            best_params = None
            best_error = float('inf')
            
            # 搜索所有组合
            for risk, sensitivity in itertools.product(
                risk_aversion_values, price_sensitivity_values
            ):
                # 运行模型
                model = run_model(risk_aversion=risk, 
                                 price_sensitivity=sensitivity)
                simulated_rate = model.get_insurance_rate()
                
                # 计算误差
                error = abs(simulated_rate - observed_insurance_rate)
                if error < best_error:
                    best_error = error
                    best_params = (risk, sensitivity)
            
            return best_params
        ```
        
        #### 3. 验证(Validation) - 预测能力评估
        
        **方法1: 历史匹配(Historical Validation)**
        ```python
        # 使用历史数据验证
        def validate_with_history():
            # 2010-2015数据用于校准
            train_data = load_data(2010, 2015)
            params = calibrate(train_data)
            
            # 2016-2020数据用于验证
            test_data = load_data(2016, 2020)
            model = MyModel(**params)
            
            # 比较模型输出与实际数据
            errors = []
            for year in range(2016, 2021):
                simulated = model.run_year(year)
                actual = test_data[year]
                error = calculate_error(simulated, actual)
                errors.append(error)
            
            return np.mean(errors), np.std(errors)
        ```
        
        **方法2: 模式结果对比**
        
        | 指标 | 实际数据 | 模型输出 | 误差 |
        |------|----------|----------|------|
        | 投保率 | 45% | 43% | -2% |
        | 平均产量 | 5.2吨/公顷 | 5.1吨/公顷 | -1.9% |
        | 价格波动率 | 0.15 | 0.18 | +20% |
        
        **方法3: 定性验证**
        - 模型是否重现关键模式(如S型扩散曲线)?
        - 政策冲击方向是否符合理论预期?
        
        #### 4. 敏感性分析
        
        **目的**: 识别关键参数,评估不确定性
        
        **单因素敏感性分析(OAT)**
        ```python
        def one_at_a_time_sensitivity():
            baseline_params = {
                'risk_aversion': 0.5,
                'price_sensitivity': 0.2,
                'social_influence': 0.3
            }
            
            results = {}
            for param_name in baseline_params.keys():
                # 固定其他参数,只变化当前参数
                param_values = np.linspace(0.1, 0.9, 9)
                outputs = []
                
                for value in param_values:
                    params = baseline_params.copy()
                    params[param_name] = value
                    model = run_model(**params)
                    output = model.get_key_metric()
                    outputs.append(output)
                
                # 计算敏感性指数
                sensitivity = np.std(outputs) / np.mean(outputs)
                results[param_name] = sensitivity
            
            return results
        ```
        
        **全局敏感性分析(Sobol)**
        - 考虑参数之间的交互作用
        - 需要使用SALib等工具
        
        #### 5. 实践建议
        
        **验证流程**:
        ```
        1. 代码验证 → 确保逻辑正确
        2. 极端测试 → 检验边界行为
        3. 参数校准 → 匹配实际数据
        4. 预测验证 → 测试预测能力
        5. 敏感性分析 → 识别关键因素
        ```
        
        **常见错误**:
        - ✗ 过度拟合(Overfitting): 参数过多,仅匹配历史
        - ✗ 忽略验证: 只校准不验证,无法评估预测能力
        - ✗ 单次运行: 忽略随机性,需要多次重复
        """,
        "key_concepts": [
            "验证Verification: 检验模型实现是否正确",
            "校准Calibration: 调整参数匹配现实数据",
            "验证Validation: 评估模型预测能力",
            "敏感性分析: 识别对结果影响最大的参数"
        ]
    },
    
    # 第三章 经典案例
    "3.1 森林火灾传播模型": {
        "learning_objective": """
        **学习目标:**
        - 理解元胞自动机与空间传播机制
        - 掌握相变现象与临界阈值
        - 学会使用网格空间建模
        """,
        "theory": """
        ### 模型原理
        
        **研究问题**: 森林密度如何影响火灾的传播范围?
        
        #### 1. 模型设定
        
        **环境**:
        - N×N网格,每个网格代表一块土地
        - 每块土地有三种状态: 空地/树木/燃烧
        
        **初始化**:
        - 按概率 p 随机生成树木
        - 最左侧一列的树木开始燃烧
        
        **传播规则**:
        - 燃烧的树木会点燃四个邻居(上下左右)的树木
        - 燃烧一步后变为空地
        
        #### 2. 相变现象(Phase Transition)
        
        **关键发现**:
        - 当树木密度 < 0.59 时: 火灾无法传播到右侧
        - 当树木密度 > 0.59 时: 火灾可以传播到右侧
        - **临界阈值 p_c ≈ 0.59** 是系统的相变点
        
        **物理类比**:
        - 类似于水的冰点: 低于0°C是冰,高于0°C是水
        - 森林密度就是控制“相变”的关键参数
        
        #### 3. 实际应用
        
        **森林管理**:
        - 防火带设计: 降低局部树木密度
        - 可燃物清理: 阻断传播路径
        
        **传染病控制**:
        - 类似机制适用于疫情传播
        - “社交距离” = 降低接触密度
        
        #### 4. 代码实现框架
        
        ```python
        class ForestFire:
            EMPTY = 0
            TREE = 1
            FIRE = 2
            
            def __init__(self, size, density):
                self.size = size
                self.grid = np.random.choice(
                    [EMPTY, TREE], 
                    size=(size, size),
                    p=[1-density, density]
                )
                # 最左侧点火
                self.grid[self.grid[:, 0] == TREE, 0] = FIRE
            
            def step(self):
                new_grid = self.grid.copy()
                fire_cells = np.argwhere(self.grid == FIRE)
                
                for i, j in fire_cells:
                    # 点燃四个邻居
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ni, nj = i+di, j+dj
                        if 0 <= ni < self.size and 0 <= nj < self.size:
                            if new_grid[ni, nj] == TREE:
                                new_grid[ni, nj] = FIRE
                    # 燃烧后变空地
                    new_grid[i, j] = EMPTY
                
                self.grid = new_grid
        ```
        """,
        "key_concepts": [
            "元胞自动机: 基于局部规则的空间演化",
            "相变现象: 系统在临界点突变",
            "临界阈值: 森林密度p≈0.59时发生相变",
            "空间传播: 通过邻域依次扩散"
        ]
    },
    
    "3.2 病毒传播模型": {
        "learning_objective": """
        **学习目标:**
        - 掌握SIR传染病模型框架
        - 理解基本传染数R0的作用
        - 学会建模群体免疫机制
        """,
        "theory": """
        ### SIR模型原理
        
        **研究问题**: 疫情如何在人群中传播?接种率多高才能阻止爆发?
        
        #### 1. 三种状态
        
        - **S (Susceptible)**: 易感者 - 健康但可能被感染
        - **I (Infected)**: 感染者 - 已感染且具传染性
        - **R (Recovered)**: 康复者 - 康复并获得免疫力
        
        #### 2. 转换规则
        
        ```
        S → I: 易感者接触感染者,有概率β被感染
        I → R: 感染者经过1/γ天后康复
        ```
        
        **参数**:
        - β: 传染率 (单次接触被感染的概率)
        - γ: 康复率 (=1/平均感染期)
        
        #### 3. 基本传染数 R0
        
        **定义**: 一个感染者在完全易感人群中平均传染的人数
        
        ```
        R0 = β / γ × 平均接触人数
        ```
        
        **临界条件**:
        - R0 < 1: 疫情自然消退
        - R0 > 1: 疫情爆发
        - R0 = 1: 临界点
        
        #### 4. 群体免疫阈值
        
        **公式**:
        ```
        群体免疫阈值 = 1 - 1/R0
        ```
        
        **示例**:
        | 疾病 | R0 | 需要接种率 |
        |------|----|-----------|
        | 新冠 | 3 | 67% |
        | 麻疹 | 15 | 93% |
        | 流感 | 1.5 | 33% |
        
        #### 5. ABM与方程模型对比
        
        **微分方程(ODE)模型**:
        ```
        dS/dt = -β·S·I/N
        dI/dt = β·S·I/N - γ·I
        dR/dt = γ·I
        ```
        
        **ABM优势**:
        - 可以建模空间网络(谁和谁接触)
        - 可以引入异质性(年龄、职业差异)
        - 可以模拟干预措施(隔离、封城)
        
        #### 6. 代码实现
        
        ```python
        class PersonAgent:
            def __init__(self, unique_id, model):
                super().__init__(unique_id, model)
                self.state = 'S'  # S/I/R
                self.infection_time = 0
            
            def step(self):
                if self.state == 'I':
                    # 康复检查
                    self.infection_time += 1
                    if random.random() < self.model.gamma:
                        self.state = 'R'
                    
                    # 传染邻居
                    neighbors = self.model.grid.get_neighbors(self.pos)
                    for neighbor in neighbors:
                        if neighbor.state == 'S':
                            if random.random() < self.model.beta:
                                neighbor.state = 'I'
        ```
        
        #### 7. 政策仿真
        
        **场景1: 社交距离**
        - 减少平均接触人数 → 降低R0
        
        **场景2: 疑似隔离**
        - 提高康复率γ (短感染期) → 降低R0
        
        **场景3: 疫苗接种**
        - 减少易感者S比例 → 当S<阈值时无法爆发
        """,
        "key_concepts": [
            "SIR模型: 易感-感染-康复三状态框架",
            "基本传染数R0: 决定疫情是否爆发的关键指标",
            "群体免疫: 足够高的接种率可阻止传播",
            "网络结构: 社交网络影响传播速度与范围"
        ]
    },
    
    "3.3 Schelling居住隔离模型": {
        "learning_objective": """
        **学习目标:**
        - 理解微观动机与宏观模式的非线性关系
        - 掌握非意图后果(unintended consequence)
        - 学会使用空间动态建模
        """,
        "theory": """
        ### 模型背景
        
        **研究问题**: 为什么微弱的个人偏好会导致强烈的社会隔离?
        
        **提出者**: Thomas Schelling (2005诺贝尔经济学奖得主)
        
        #### 1. 模型设计
        
        **智能体**:
        - 两种类型: 红色和蓝色(或X和O)
        - 每个智能体占据网格的一个位置
        
        **决策规则**:
        ```python
        # 观察邻居(八个方向)
        similar_neighbors = 同类型邻居数量
        total_neighbors = 总邻居数量
        
        similarity_ratio = similar_neighbors / total_neighbors
        
        # 判断是否满意
        if similarity_ratio >= threshold:  # 例如 30%
            stay  # 留下
        else:
            move_to_empty  # 移动到空格子
        ```
        
        **关键参数**:
        - `threshold`: 满意阈值(通常0.3即可)
        
        #### 2. 核心发现
        
        **反直觉结果**:
        
        | 个人偏好 | 宏观结果 |
        |----------|----------|
        | 阈值=30% (微弱) | 隔离率>70% (强烈) |
        | “只要不全是外族就行” | “实际形成高度隔离社区” |
        
        **为什么会这样?**
        
        1. **正反馈循环**:
           - 少数人移出 → 区域更单一 → 更多人移出
        
        2. **雪崩效应**:
           - 初始小扰动 → 逐步放大 → 最终完全隔离
        
        3. **局部优化≠全局最优**:
           - 每个人都在为自己找更好的位置
           - 但集体结果却是大家都不想要的隔离
        
        #### 3. 政策含义
        
        **住房政策**:
        - 单纯反歧视法律不足以消除隔离
        - 需要积极干预(如混合社区建设)
        
        **教育政策**:
        - 学校选择自由可能加剧隔离
        - 需要划片区或配额制度
        
        **金融市场**:
        - 类似机制解释市场分层现象
        - 微弱偏好→市场分割
        
        #### 4. 代码实现
        
        ```python
        class SchellingAgent(Agent):
            def __init__(self, unique_id, model, agent_type):
                super().__init__(unique_id, model)
                self.type = agent_type  # 0或1
                self.happy = False
            
            def step(self):
                # 获取邻居
                neighbors = self.model.grid.get_neighbors(
                    self.pos, 
                    moore=True,  # 八邻居
                    include_center=False
                )
                
                if len(neighbors) == 0:
                    self.happy = True
                    return
                
                # 计算同类比例
                similar = sum(1 for n in neighbors if n.type == self.type)
                similarity = similar / len(neighbors)
                
                # 判断满意度
                self.happy = similarity >= self.model.homophily
                
                # 不满意则移动
                if not self.happy:
                    self.model.grid.move_to_empty(self)
        ```
        
        #### 5. 扩展实验
        
        **变量1: 改变阈值**
        - threshold = 0.2: 较快达到中度隔离
        - threshold = 0.5: 需要更长时间,隔离更强
        
        **变量2: 初始分布**
        - 随机分布: 最终一定隔离
        - 均匀混合: 同样会隔离
        
        **变量3: 网络结构**
        - 网格空间: 形成聚集区
        - 网络图: 形成社群
        
        #### 6. 理论启示
        
        **核心洞见**: 
        > 微观动机与宏观模式之间存在巨大鸿沟,个人的温和偏好可以导致社会的极端结果。
        
        **方法论价值**:
        - 证明ABM在揭示复杂系统涌现性上的独特作用
        - 成为ABM领域最经典的教学案例
        """,
        "key_concepts": [
            "微观动机与宏观结果: 弱偏好导致强隔离",
            "非意图后果: 个体理性不保证集体最优",
            "正反馈循环: 小变化被放大成大变化",
            "政策干预: 需要积极干预才能改变隔离模式"
        ]
    },
    
    "3.4 技术扩散模型": {
        "learning_objective": """
        **学习目标:**
        - 掌握Rogers创新扩散理论
        - 理解S型曲线的形成机制
        - 学会建模社会学习过程
        """,
        "theory": """
        ### Rogers创新扩散理论
        
        **研究问题**: 新技术如何在社会中扩散?为什么呈现S型曲线?
        
        #### 1. 采纳者分类
        
        **五种类型** (按采纳时间排序):
        
        | 类型 | 比例 | 特征 |
        |------|------|------|
        | **创新者** | 2.5% | 风险偏好高,资源丰富 |
        | **早期采纳者** | 13.5% | 意见领袖,社交中心 |
        | **早期多数** | 34% | 等待观望,跟随趋势 |
        | **晚期多数** | 34% | 谨慎保守,资源有限 |
        | **落后者** | 16% | 抗拒变化,隔离信息 |
        
        #### 2. S型扩散曲线
        
        ```
        采纳率
          ^
        100%|　　　　　　　___---
          |　　　　　__---
          |　　　__--  【快速增长期】
          |　　_-
          |__-  【缓慢起步期】
          +------------------------> 时间
        ```
        
        **三个阶段**:
        1. **缓慢起步**: 只有创新者采纳
        2. **快速增长**: 群体效应,爆发式增长
        3. **趋于饱和**: 剩余落后者,增速放缓
        
        #### 3. ABM建模机制
        
        **决策规则** (多因素综合):
        
        ```python
        def decide_adoption(self):
            # 1. 个人特征
            personal_tendency = self.innovativeness  # 0-1
            
            # 2. 社会影响
            neighbors = self.get_neighbors()
            adoption_rate = sum([n.adopted for n in neighbors]) / len(neighbors)
            social_influence = adoption_rate * self.social_weight
            
            # 3. 技术属性
            perceived_benefit = self.calc_benefit()
            perceived_cost = self.calc_cost()
            
            # 4. 综合决策
            adoption_prob = (
                personal_tendency * 0.3 +
                social_influence * 0.4 +
                (perceived_benefit - perceived_cost) * 0.3
            )
            
            return random.random() < adoption_prob
        ```
        
        **关键参数**:
        - `innovativeness`: 个体创新性 (0-1)
        - `social_weight`: 社会影响权重
        - `network_structure`: 网络拓扑(小世界/随机/无标度)
        
        #### 4. 网络结构的影响
        
        | 网络类型 | 扩散速度 | 特点 |
        |----------|----------|------|
        | **随机网络** | 中等 | 均匀扩散 |
        | **小世界网络** | 最快 | 局部聚集+远程连接 |
        | **无标度网络** | 分化 | 枢纽节点加速扩散 |
        | **格子网络** | 最慢 | 只有局部传播 |
        
        #### 5. 政策仿真应用
        
        **农业技术推广**:
        ```python
        # 场景1: 找到早期采纳者
        # 通过网络中心性识别意见领袖
        opinion_leaders = find_high_centrality_nodes(network)
        for leader in opinion_leaders:
            leader.receive_training()  # 重点培训
        
        # 场景2: 降低采纳门槛
        technology.cost = technology.cost * 0.7  # 补贴
        technology.ease_of_use = 0.8  # 提供培训
        
        # 场景3: 可见性建设
        # 设置示范户,增加社会影响权重
        demo_farmers = select_demo_sites()
        for farmer in demo_farmers:
            farmer.visibility = 1.0  # 高可见度
        ```
        
        **金融产品创新**:
        - 分析移动支付的扩散
        - 评估数字货币的接受度
        
        #### 6. 代码框架
        
        ```python
        class FarmerAgent(Agent):
            def __init__(self, unique_id, model):
                super().__init__(unique_id, model)
                # 个体特征
                self.innovativeness = np.random.beta(2, 5)  # 偏保守
                self.wealth = np.random.lognormal(10, 1)
                self.education = random.choice(['low', 'medium', 'high'])
                
                # 采纳状态
                self.adopted = False
                self.adoption_time = None
            
            def step(self):
                if not self.adopted:
                    # 观察邻居
                    neighbors = self.model.grid.get_neighbors(self.pos)
                    adopted_neighbors = [n for n in neighbors if n.adopted]
                    
                    # 计算采纳概率
                    prob = self.calc_adoption_probability(
                        len(adopted_neighbors) / len(neighbors)
                    )
                    
                    # 决策
                    if random.random() < prob:
                        self.adopted = True
                        self.adoption_time = self.model.schedule.time
        ```
        
        #### 7. 实验设计
        
        **变量控制**:
        - 独立变量: 社会影响权重
        - 依赖变量: 达到50%采纳率的时间
        
        **结果评估**:
        - 绘制S型曲线
        - 计算扩散速度
        - 分析网络效应
        """,
        "key_concepts": [
            "S型曲线: 创新扩散的典型模式",
            "社会学习: 通过观察邻居做出决策",
            "早期采纳者: 意见领袖在扩散中的关键作用",
            "网络拓扑: 小世界网络最利于技术扩散"
        ]
    },
    
    # 第四章 金融ABM
    "4.1 银行风险传染模型": {
        "learning_objective": """
        **学习目标:**
        - 理解银行间风险传染机制
        - 掌握复杂网络在金融传染中的作用
        - 学会建模系统性风险
        """,
        "theory": """
        ### 模型背景
        
        **研究问题**: 2008年金融危机中,为什么少数银行的问题会导致系统性崩溃?
        
        #### 1. 传染渠道
        
        **直接暴露(Direct Exposure)**:
        ```python
        # 银行A对银行B有借款
        if bank_B.defaults:
            bank_A.loss += bank_A.exposure_to_B
            if bank_A.loss > bank_A.capital:
                bank_A.defaults = True  # A也违约
        ```
        
        **流动性冲击(Liquidity Shock)**:
        ```python
        # 银行A需要紧急出售资产
        fire_sale_loss = asset_value * discount_rate
        # 降价抛售导致资产价格下跌
        # 其他银行持有的类似资产也贬值
        for other_bank in banks:
            other_bank.asset_value -= spillover_effect
        ```
        
        **信心传染(Confidence Contagion)**:
        ```python
        # 一家银行出问题→存款人恐慌
        if bank_neighbor.defaults:
            self.deposit_withdrawal_rate += panic_factor
        ```
        
        #### 2. 网络结构的影响
        
        | 网络类型 | 特征 | 风险特点 |
        |----------|------|----------|
        | **随机网络** | 连接随机分布 | 风险分散,但普遍传染 |
        | **无标度网络** | 少数枢纽节点 | 枢纽银行失败→系统崩溃 |
        | **核心-边缘** | 大银行为核心 | “大而不能倒”问题 |
        
        **关键发现**: 无标度网络最脆弱,因为少数关键节点集中了大量连接。
        
        #### 3. 系统性风险指标
        
        ```python
        # 违约比例
        default_rate = 违约银行数 / 总银行数
        
        # 网络传染率
        contagion_rate = 被传染银行数 / 初始冲击银行数
        
        # 资本损失
        total_loss = sum([bank.capital_loss for bank in banks])
        ```
        
        #### 4. 政策仿真
        
        **场景1: 资本充足率要求**
        ```python
        # Basel III: 要求资本充足率>8%
        for bank in banks:
            if bank.capital / bank.assets < 0.08:
                bank.increase_capital()
        ```
        
        **场景2: 中央银行救助**
        ```python
        # 对系统重要性银行注资
        for bank in systemically_important_banks:
            if bank.at_risk:
                central_bank.inject_capital(bank)
        ```
        """,
        "key_concepts": [
            "直接暴露: 银行间借贷关系导致连锁违约",
            "流动性传染: 抛售资产导致价格螺旋下降",
            "系统重要性: 枢纽节点失败影响整个系统",
            "宏观审慎监管: 关注网络结构和系统性风险"
        ]
    },
    
    "4.2 信贷市场定价模型": {
        "learning_objective": """
        **学习目标:**
        - 理解Stiglitz-Weiss信贷配给理论
        - 掌握逆向选择机制
        - 学会建模利率定价问题
        """,
        "theory": """
        ### 信贷配给问题
        
        **研究问题**: 为什么银行不通过提高利率来清空市场?
        
        #### 1. 逆向选择机制
        
        **传统观点** (错误):
        ```
        利率↑ → 银行收益↑
        ```
        
        **Stiglitz-Weiss洞见**:
        ```
        利率↑ → 低风险借款人退出 → 平均风险↑ → 银行收益↓
        ```
        
        #### 2. 数学模型
        
        **银行期望收益**:
        ```
        E[π] = (1-p)·r·L - p·L
        ```
        其中: r=利率, L=贷款额, p=违约概率
        
        **关键**: p 是 r 的函数
        - r 低时: 各类借款人都申请, p 中等
        - r 高时: 低风险者退出, p 上升
        
        **结果**: 存在最优利率 r*, 使 E[π] 最大化
        
        #### 3. ABM实现
        
        ```python
        class BorrowerAgent:
            def __init__(self, risk_type):
                self.risk_type = risk_type  # 'low', 'medium', 'high'
                self.default_prob = {'low': 0.05, 'medium': 0.15, 'high': 0.30}
                self.expected_return = {'low': 0.08, 'medium': 0.12, 'high': 0.18}
            
            def decide_apply(self, interest_rate):
                # 只有期望收益>利率才申请
                return self.expected_return[self.risk_type] > interest_rate
        
        class BankAgent:
            def set_interest_rate(self, rate):
                self.rate = rate
                # 统计申请者池
                applicants = [b for b in borrowers if b.decide_apply(rate)]
                # 计算平均违约率
                avg_default = np.mean([b.default_prob[b.risk_type] 
                                      for b in applicants])
                # 计算期望收益
                return (1 - avg_default) * rate - avg_default
        ```
        
        #### 4. 政策启示
        
        **信贷配给解决方案**:
        - 抵押品要求
        - 信用评分系统
        - 分层定价(差别化利率)
        - 关系型借贷
        """,
        "key_concepts": [
            "逆向选择: 高利率驱逐优质借款人",
            "利率惖论: 利率过高反而降低银行收益",
            "信贷配给: 市场均衡时仍有贷款需求未满足",
            "差别化定价: 根据风险分层设定不同利率"
        ]
    },
    
    "4.3 激励机制设计模型": {
        "learning_objective": """
        **学习目标:**
        - 理解委托-代理问题
        - 掌握动态激励机制
        - 学会建模学习适应过程
        """,
        "theory": """
        ### 委托-代理框架
        
        **研究问题**: 如何设计激励合同促使代理人努力工作?
        
        #### 1. 基本问题
        
        **信息不对称**:
        - 代理人的努力不可观测
        - 结果受努力+运气影响
        
        **利益冲突**:
        - 委托人: 希望最大化产出
        - 代理人: 希望最小化努力成本
        
        #### 2. 激励合同设计
        
        **固定工资 vs 业绩奖金**:
        ```python
        # 方案A: 纯固定工资
        payment_A = base_salary  # 代理人不努力
        
        # 方案B: 业绩奖金
        payment_B = base + bonus_rate * performance
        # 代理人有动力努力,但承担风险
        ```
        
        **最优激励强度**:
        ```
        激励强度 = f(努力效率, 风险厌恶, 运气成分)
        ```
        
        #### 3. 动态学习模型
        
        ```python
        class PrincipalAgent:
            def __init__(self):
                # 委托人使用强化学习调整激励系数
                self.bonus_rate = 0.5
                self.q_table = {}  # 状态-动作-价值
            
            def update_incentive(self, performance, cost):
                # 根据上期表现调整
                if performance > target:
                    self.bonus_rate *= 0.95  # 降低激励
                else:
                    self.bonus_rate *= 1.05  # 提高激励
        
        class AgentWorker:
            def choose_effort(self, bonus_rate):
                # 代理人根据激励决定努力程度
                expected_benefit = bonus_rate * expected_performance
                effort_cost = self.calc_effort_cost(effort_level)
                
                # 选择最优努力水平
                optimal_effort = argmax(expected_benefit - effort_cost)
                return optimal_effort
        ```
        
        #### 4. 应用场景
        
        **金融机构**:
        - 经理人薪酬设计
        - 分析师业绩考核
        
        **保险合同**:
        - 免赔额设计
        - 理赔效率激励
        """,
        "key_concepts": [
            "委托-代理问题: 信息不对称导致激励不兼容",
            "动态激励: 根据表现反馈调整激励强度",
            "强化学习: 智能体通过试错寻找最优策略",
            "风险分担: 平衡激励强度与风险转移"
        ]
    },
    
    # 第五章 农业金融ABM
    "5.1 农户作物选择模型": {
        "learning_objective": """
        **学习目标:**
        - 掌握农户风险决策框架
        - 理解价格信号与作物结构
        - 学会建模邻里学习效应
        """,
        "theory": """
        ### 农户决策框架
        
        **研究问题**: 价格波动如何影响农户的作物种植结构?
        
        #### 1. 两种作物选择
        
        **玉米 vs 大豆**:
        ```python
        # 期望收益
        profit_corn = price_corn * yield_corn - cost_corn
        profit_soybean = price_soybean * yield_soybean - cost_soybean
        
        # 考虑风险
        utility_corn = E[profit] - risk_aversion * Var[profit]
        ```
        
        #### 2. 社会学习机制
        
        ```python
        def decide_crop_with_learning(self):
            # 个人判断
            personal_choice = self.calc_optimal_crop()
            
            # 观察邻居
            neighbors = self.get_neighbors()
            neighbor_choices = [n.last_crop for n in neighbors]
            popular_crop = mode(neighbor_choices)
            
            # 加权综合
            if random.random() < self.social_learning_weight:
                return popular_crop
            else:
                return personal_choice
        ```
        """,
        "key_concepts": [
            "风险厌恶: 农户偏好稳定收益而非高波动收益",
            "邻里学习: 通过观察邻居决策获取信息",
            "价格传导: 市场价格影响种植结构调整",
            "路径依赖: 历史种植经验影响当前选择"
        ]
    },
    
    "5.2 农业保险采纳模型": {
        "learning_objective": """
        **学习目标:**
        - 掌握期望效用理论在保险决策中的应用
        - 理解政府补贴对投保率的影响
        - 学会评估保险的福利效应
        """,
        "theory": """
        ### 保险决策模型
        
        #### 1. 期望效用框架
        
        ```python
        # 无保险
        EU_no_ins = (1-p) * U(W) + p * U(W - Loss)
        
        # 有保险
        EU_with_ins = (1-p) * U(W - Premium) + \
                      p * U(W - Premium - Loss + Payout)
        
        # 决策
        if EU_with_ins > EU_no_ins:
            buy_insurance()
        ```
        
        #### 2. 政策补贴影响
        
        ```python
        # 补贴前
        premium_full = actuarially_fair_premium
        
        # 补贴后(例如60%)
        premium_farmer_pays = premium_full * 0.4
        
        # 投保率显著提高
        ```
        """,
        "key_concepts": [
            "期望效用: 综合考虑概率和效用的决策框架",
            "风险厌恶: 凹效用函数体现对风险的厌恶",
            "政府补贴: 降低农户保费负担提高投保率",
            "福利效应: 保险减少收入波动提高农户福利"
        ]
    },
    
    "5.3 农村信贷风险模型": {
        "learning_objective": """
        **学习目标:**
        - 理解农户信用评级与违约风险
        - 掌握银保联动机制
        - 学会建模信贷配给问题
        """,
        "theory": """
        ### 农村信贷模型
        
        #### 1. 信用分层
        
        ```python
        class FarmerBorrower:
            def calc_credit_score(self):
                score = (
                    self.wealth * 0.3 +
                    self.land_area * 0.2 +
                    self.education * 0.2 +
                    self.credit_history * 0.3
                )
                return score
            
            def get_loan_terms(self, score):
                if score > 0.8:
                    return {'rate': 0.05, 'amount': 0.8 * collateral}
                elif score > 0.5:
                    return {'rate': 0.08, 'amount': 0.5 * collateral}
                else:
                    return {'rate': 0.12, 'amount': 0.3 * collateral}
        ```
        
        #### 2. 银保联动
        
        ```python
        # 有保险的农户违约风险降低
        if farmer.has_insurance:
            default_prob *= (1 - insurance_coverage)
            # 银行给予更优惠利率
            interest_rate -= 0.01
        ```
        """,
        "key_concepts": [
            "信用评级: 基于多维信息评估违约风险",
            "分层定价: 不同风险级别对应不同利率",
            "银保联动: 保险降低信贷风险改善融资条件",
            "抵押品: 土地经营权作为信贷抵押"
        ]
    },
    
    "5.4 粮食市场综合模型": {
        "learning_objective": """
        **学习目标:**
        - 掌握多主体交互模型构建
        - 理解政策组合效应
        - 学会评估粮食安全与财政可持续性
        """,
        "theory": """
        ### 综合模型框架
        
        #### 1. 七类主体
        
        1. **农户**: 作物选择+投保+贷款决策
        2. **收购商**: 国企粮库/加工企业/贸易商
        3. **保险公司**: 保费计算+理赔
        4. **银行**: 信贷评估+还款管理
        5. **政府**: 关税+补贴+托市价
        6. **国外部门**: 进口+出口
        7. **环境**: 灾害+价格冲击
        
        #### 2. 核心机制
        
        **价格形成**:
        ```python
        # 国内价格 = f(供给, 需求, 库存, 关税, 托市价)
        supply = sum([f.output for f in farmers])
        demand = population * per_capita_consumption
        
        if supply < demand:
            import_quantity = demand - supply
            price = world_price * (1 + tariff_rate)
        else:
            price = max(floor_price, market_clearing_price)
        ```
        
        #### 3. 政策评估
        
        **三重目标**:
        1. 粮食自给率 ≥ 阈值(如 95%)
        2. 农户收入波动 ≤ 可接受范围
        3. 财政支出 ≤ 预算上限
        """,
        "key_concepts": [
            "多主体交互: 7类主体共同决定市场结果",
            "政策组合: 关税+保险+信贷+补贴的协同效应",
            "财政约束: 在预算约束下平衡多重目标",
            "情景模拟: 测试不同政策组合的稳健性"
        ]
    }
}

# ============================================================================
# 简单实验案例
# ============================================================================

class SimpleAgent(Agent):
    """简单智能体示例"""
    def __init__(self, model):
        super().__init__(model)
        self.wealth = np.random.uniform(50, 150)
        self.cooperate = np.random.choice([True, False])
    
    def step(self):
        # 简单的财富增长规则
        if self.cooperate:
            self.wealth += np.random.uniform(0, 5)
        else:
            self.wealth += np.random.uniform(-2, 8)

class SimpleModel(Model):
    """简单ABM模型示例"""
    def __init__(self, n_agents=50):
        super().__init__()
        self.n_agents = n_agents
        
        # 创建智能体
        for i in range(self.n_agents):
            agent = SimpleAgent(self)
        
        # 数据收集
        self.datacollector = DataCollector(
            model_reporters={
                "平均财富": lambda m: np.mean([a.wealth for a in m.agents]),
                "合作比例": lambda m: np.mean([a.cooperate for a in m.agents])
            }
        )
    
    def step(self):
        self.datacollector.collect(self)
        # Mesa 3.x使用agents列表
        for agent in self.agents:
            agent.step()


# ============================================================================
# 第三章 经典案例实验模型
# ============================================================================

class ForestFireModel(Model):
    """森林火灾传播模型"""
    def __init__(self, width=50, height=50, density=0.6):
        super().__init__()
        self.width = width
        self.height = height
        self.density = density
        
        # 初始化网格: 0=空地, 1=树木, 2=燃烧
        self.grid = np.random.choice([0, 1], size=(height, width), 
                                      p=[1-density, density])
        # 最左侧点火
        self.grid[self.grid[:, 0] == 1, 0] = 2
        
        self.datacollector = DataCollector(
            model_reporters={
                "Trees": lambda m: np.sum(m.grid == 1),
                "Fire": lambda m: np.sum(m.grid == 2),
                "Burned": lambda m: np.sum(m.grid == 0)
            }
        )
        self.datacollector.collect(self)
    
    def step(self):
        new_grid = self.grid.copy()
        fire_cells = np.argwhere(self.grid == 2)
        
        for i, j in fire_cells:
            # 点燃四个邻居
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < self.height and 0 <= nj < self.width:
                    if new_grid[ni, nj] == 1:
                        new_grid[ni, nj] = 2
            # 燃烧后变空地
            new_grid[i, j] = 0
        
        self.grid = new_grid
        self.datacollector.collect(self)


class SIRAgent(Agent):
    """病毒传播模型SIR智能体"""
    def __init__(self, model, pos):
        super().__init__(model)
        self.pos = pos
        self.state = "S"  # S/I/R
        self.infection_time = 0
    
    def step(self):
        if self.state == "I":
            self.infection_time += 1
            # 康复检查
            if np.random.random() < self.model.gamma:
                self.state = "R"
                return
            
            # 传染邻居
            neighbors = self.get_neighbors()
            for neighbor in neighbors:
                if neighbor.state == "S":
                    if np.random.random() < self.model.beta:
                        neighbor.state = "I"
    
    def get_neighbors(self):
        """Moore邻域(八邻居)"""
        neighbors = []
        x, y = self.pos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = (x + dx) % self.model.width, (y + dy) % self.model.height
                if self.model.grid[nx][ny] is not None:
                    neighbors.append(self.model.grid[nx][ny])
        return neighbors


class SIRModel(Model):
    """病毒传播模型"""
    def __init__(self, width=50, height=50, density=0.9, beta=0.3, gamma=0.1, initial_infected=5):
        super().__init__()
        self.width = width
        self.height = height
        self.beta = beta  # 传染率
        self.gamma = gamma  # 康复率
        
        # 初始化网格
        self.grid = [[None for _ in range(width)] for _ in range(height)]
        
        # 生成智能体
        positions = [(i, j) for i in range(height) for j in range(width)]
        positions = np.random.choice(len(positions), int(width * height * density), replace=False)
        
        agent_list = []
        for idx in positions:
            i, j = idx // width, idx % width
            agent = SIRAgent(self, (i, j))
            self.grid[i][j] = agent
            agent_list.append(agent)
        
        # 初始感染
        infected_agents = np.random.choice(agent_list, initial_infected, replace=False)
        for agent in infected_agents:
            agent.state = "I"
        
        self.datacollector = DataCollector(
            model_reporters={
                "Susceptible": lambda m: sum(1 for a in m.agents if a.state == "S"),
                "Infected": lambda m: sum(1 for a in m.agents if a.state == "I"),
                "Recovered": lambda m: sum(1 for a in m.agents if a.state == "R")
            }
        )
        self.datacollector.collect(self)
    
    def step(self):
        for agent in self.agents:
            agent.step()
        self.datacollector.collect(self)


class SchellingAgent(Agent):
    """Schelling隔离模型智能体"""
    def __init__(self, model, pos, agent_type):
        super().__init__(model)
        self.pos = pos
        self.type = agent_type  # 0或1
        self.happy = False
    
    def step(self):
        neighbors = self.get_neighbors()
        if len(neighbors) == 0:
            self.happy = True
            return
        
        # 计算同类比例
        similar = sum(1 for n in neighbors if n.type == self.type)
        similarity = similar / len(neighbors)
        
        # 判断满意度
        self.happy = similarity >= self.model.homophily
        
        # 不满意则移动
        if not self.happy:
            self.move_to_empty()
    
    def get_neighbors(self):
        """Moore邻域"""
        neighbors = []
        x, y = self.pos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx = (x + dx) % self.model.width
                ny = (y + dy) % self.model.height
                if self.model.grid[nx][ny] is not None:
                    neighbors.append(self.model.grid[nx][ny])
        return neighbors
    
    def move_to_empty(self):
        """移动到随机空位置"""
        empty_cells = [(i, j) for i in range(self.model.width) 
                       for j in range(self.model.height) 
                       if self.model.grid[i][j] is None]
        if empty_cells:
            # 清空当前位置
            old_x, old_y = self.pos
            self.model.grid[old_x][old_y] = None
            # 移动到新位置
            new_pos = empty_cells[np.random.randint(len(empty_cells))]
            self.pos = new_pos
            self.model.grid[new_pos[0]][new_pos[1]] = self


class SchellingModel(Model):
    """Schelling居住隔离模型"""
    def __init__(self, width=50, height=50, density=0.8, homophily=0.3, minority_pc=0.5):
        super().__init__()
        self.width = width
        self.height = height
        self.homophily = homophily  # 满意阈值
        
        # 初始化网格
        self.grid = [[None for _ in range(width)] for _ in range(height)]
        
        # 生成智能体
        n_agents = int(width * height * density)
        positions = [(i, j) for i in range(height) for j in range(width)]
        np.random.shuffle(positions)
        
        for idx in range(n_agents):
            i, j = positions[idx]
            agent_type = 0 if np.random.random() < minority_pc else 1
            agent = SchellingAgent(self, (i, j), agent_type)
            self.grid[i][j] = agent
        
        self.datacollector = DataCollector(
            model_reporters={
                "Happy": lambda m: sum(1 for a in m.agents if a.happy),
                "Unhappy": lambda m: sum(1 for a in m.agents if not a.happy),
                "Segregation": self.calculate_segregation
            }
        )
        self.datacollector.collect(self)
    
    def calculate_segregation(self):
        """计算隔离程度"""
        total_similarity = 0
        count = 0
        for agent in self.agents:
            neighbors = agent.get_neighbors()
            if len(neighbors) > 0:
                similar = sum(1 for n in neighbors if n.type == agent.type)
                total_similarity += similar / len(neighbors)
                count += 1
        return total_similarity / count if count > 0 else 0
    
    def step(self):
        # 随机打乱顺序
        agents = list(self.agents)
        np.random.shuffle(agents)
        for agent in agents:
            agent.step()
        self.datacollector.collect(self)

# ============================================================================
# 推荐阅读与学习资源
# ============================================================================
RECOMMENDED_READINGS = {
    "金融ABM核心文献": [
        "张亮. 复杂性视角下银行体系风险传染的计算实验研究[D]. 河北工业大学博士学位论文, 2017.",
        "马泽宇. 基于计算实验的银行间市场风险传染研究[D]. 天津大学硕士学位论文, 2015.",
        "熊熊, 郭翠, 张维, 张永杰. 中小企业贷款利率定价的计算实验方法[J]. 系统工程理论与实践, 2010.",
        "时茜茜, 姚隆玉, 李博雅, 朱建波. 基于计算实验的重大工程风险管理激励效率演化分析[J]. 系统管理学报, 2024.",
        "刘征驰. 个体认知、群体共识与互联网众筹投资绩效——基于计算实验方法的研究[J]. 2020.",
        "张维等. 计算实验金融工程：大数据驱动的金融管理决策工具[J]. 2014.",
        "苟梦颖. 金融研究的新范式：计算实验金融的发展及应用[J]. 2015.",
        "Axtell, R. L., & Farmer, J. D. (2025). Agent-based modeling in economics and finance: Past, present, and future[J]. Journal of Economic Literature.",
        "Bookstaber, R., Paddrik, M., & Tivnan, B. (2018). An agent-based model for financial vulnerability[J]. Journal of Economic Interaction and Coordination, 13(2), 433-466."
    ],
    "农业保险与风险管理": [
        "Will, M., Groeneveld, J., Frank, K., & Müller, B. (2021). Informal risk-sharing between smallholders may be threatened by formal insurance: Lessons from a stylized agent-based model[J]. PLoS ONE, 16(3): e0248757.",
        "Barnaud, C., Bousquet, F., & Trebuil, G. (2008). Multi-agent simulations to explore rules for rural credit in a highland farming community of Northern Thailand[J]. Ecological Economics, 66(4), 615-627.",
        "Dubbelboer, J., Nikolic, I., Jenkins, K., & Hall, J. (2017). An agent-based model of flood risk and insurance[J]. Journal of Artificial Societies and Social Simulation, 20(1), 6.",
        "Owadally, I., Zhou, F., Otunba, R., Lin, J., & Wright, D. (2019). An agent-based system with temporal data mining for monitoring financial stability on insurance markets[J]. Expert Systems with Applications, 123, 270-282."
    ],
    "农业技术采纳与扩散": [
        "Berger, T. (2001). Agent‐based spatial models applied to agriculture: A simulation tool for technology diffusion, resource use changes and policy analysis[J]. Agricultural Economics, 25(2‐3), 245-260.",
        "Alotibi, Y. S. (2025). A socio-technical agent-based simulation model for predicting smart agriculture adoption dynamics[J]. Scientific Reports, 15, Article 1234.",
        "Barbuto, A., Lopolito, A., & Santeramo, F. G. (2019). Improving diffusion in agriculture: An agent-based model to find the predictors for efficient early adopters[J]. Agricultural and Food Economics, 7(1), 7.",
        "Orjuela-Garzon, W., Quintero, S., Giraldo, D. P., & Lotero, L. (2021). A framework for analysing technology transfer processes using agent-based modelling: A case study on massive technology adoption (AMTEC) program on coffee farms[J]. Sustainability, 13(20), 11143.",
        "Schreinemachers, P., & Berger, T. (2009). The diffusion of greenhouse agriculture in Northern Thailand: Combining econometrics and agent‐based modeling[J]. Journal of Agricultural Economics, 40(4), 373-388.",
        "De Keyser, E., Farahbakhsh, S., & Mathijs, E. (2025). Farmers' decision-making dynamics in bio-based fertilizer adoption: An agent-based model[J]. Agricultural and Food Economics, 13(1), 1-24."
    ],
    "农业政策评估": [
        "Kremmydas, D., Athanasiadis, I. N., & Rozakis, S. (2018). A review of agent based modeling for agricultural policy evaluation[J]. Agricultural Systems, 164, 95-106.",
        "Sun, R., Nolan, J., & Kulshreshtha, S. (2022). Agent-based modeling of policy induced agri-environmental technology adoption[J]. SN Business & Economics, 2(7), 78.",
        "Bazzana, D., Foltz, J., & Zhang, Y. (2022). Impact of climate smart agriculture on food security: An agent-based analysis[J]. Food Policy, 111, 102304.",
        "Berger, T., & Troost, C. (2014). Agent‐based modelling of climate adaptation and mitigation options in agriculture[J]. Journal of Agricultural Economics, 65(2), 323-348.",
        "Happe, K., Kellermann, K., & Balmann, A. (2006). Agent-based analysis of agricultural policies: An illustration of the agricultural policy simulator AgriPoliS, its adaptation and behavior[J]. Ecology and Society, 11(1), 49."
    ],
    "中国农业ABM研究": [
        "常笑, 刘黎明, 刘朝旭, 陈伟强. 农户土地利用决策行为的多智能体模拟方法[J]. 农业工程学报, 2013, 29(14): 230-241.",
        "吕晓, 牛善栋, 李振波, 黄贤金. 中国耕地集约利用研究现状及趋势分析[J]. Transactions of the Chinese Society of Agricultural Engineering, 2015.",
        "顾润男, 刘泽照. 公共政策模拟研究: 议题, 类属与趋势[J]. 西部经济管理论坛, 2023.",
        "高仙草, 任艳云, 谭贺, 孔贺, 高发瑞. 基于文献计量的农业指数保险态势分析[J]. Agricultural Outlook, 2023."
    ]
}

# ============================================================================
# 主界面构建
# ============================================================================

def main():
    # 页面标题
    st.markdown(f"<h1 class='main-title'>🎓 ABM教学实验平台</h1>", 
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>从基本原理到农业金融保险应用的完整学习路径</p>", 
                unsafe_allow_html=True)

    # 推荐阅读与学习资源栏目
    with st.expander("📚 推荐阅读与学习资源", expanded=False):
        st.markdown("**文献综述与重点中英文论文(可课后进一步查阅原文)**")
        for category, items in RECOMMENDED_READINGS.items():
            st.markdown(f"#### {category}")
            for ref in items:
                query = urllib.parse.quote(ref)
                search_url = f"https://scholar.google.com/scholar?q={query}"
                st.markdown(f"- {ref} [[DOI/全文检索]]({search_url})")

    # 侧边栏 - 学习路径导航
    with st.sidebar:
        st.markdown("## 📖 学习路径")
        
        # 字体大小设置
        st.markdown("---")
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("### ⚙️ 设置")
        with col2:
            pass
        
        font_option = st.selectbox(
            "字体大小",
            ["小", "中等", "大", "特大"],
            index=["小", "中等", "大", "特大"].index(st.session_state.font_size),
            key="font_selector"
        )
        
        if font_option != st.session_state.font_size:
            st.session_state.font_size = font_option
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📚 章节导航")
        
        for chapter, content in LEARNING_PATH.items():
            with st.expander(f"{content['icon']} {chapter}", 
                           expanded=(chapter == st.session_state.current_chapter)):
                for section in content['sections']:
                    # 显示完成状态
                    completed = st.session_state.progress.get(section, False)
                    status_icon = "✅" if completed else "⭕"
                    
                    if st.button(f"{status_icon} {section}", key=f"nav_{section}",
                               use_container_width=True):
                        st.session_state.current_chapter = chapter
                        st.session_state.current_section = section
                        st.rerun()
        
        st.markdown("---")
        
        # 学习进度统计
        total_sections = sum(len(v['sections']) for v in LEARNING_PATH.values())
        completed_sections = sum(st.session_state.progress.values())
        progress_pct = completed_sections / total_sections if total_sections > 0 else 0
        
        st.markdown("### 📊 学习进度")
        st.progress(progress_pct)
        st.markdown("已完成: {}/{} 节".format(completed_sections, total_sections))
        
        # 导出功能
        st.markdown("---")
        st.markdown("### 💾 数据管理")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("💾 保存进度", use_container_width=True):
                save_progress()
                st.success("✅ 进度已保存")
        
        with col_b:
            if st.button("📄 导出报告", use_container_width=True):
                report = generate_learning_report()
                st.download_button(
                    label="下载报告",
                    data=report,
                    file_name=f"学习报告_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
    
    # 主内容区
    display_section_content(st.session_state.current_section)

def display_section_content(section: str):
    """显示章节内容"""
    
    # 章节标题
    st.markdown(f"<div class='chapter-title'>{section}</div>", unsafe_allow_html=True)
    
    # 检查是否有理论内容
    if section in THEORY_CONTENT:
        content = THEORY_CONTENT[section]
        
        # 学习目标
        st.markdown(f"<div class='learning-objective'>{content['learning_objective']}</div>", 
                   unsafe_allow_html=True)
        
        # 理论讲解
        st.markdown(f"<div class='theory-box'>{content['theory']}</div>", 
                   unsafe_allow_html=True)
        
        # 关键概念
        if 'key_concepts' in content:
            st.markdown("### 🔑 关键概念")
            for concept in content['key_concepts']:
                st.markdown(f"<div class='key-concept'>💡 {concept}</div>", 
                          unsafe_allow_html=True)
        
        # 交互实验部分(如果有)
        if section == "1.1 什么是基于智能体建模":
            display_simple_experiment()
        elif section == "3.1 森林火灾传播模型":
            display_forest_fire_experiment()
        elif section == "3.2 病毒传播模型":
            display_sir_experiment()
        elif section == "3.3 Schelling居住隔离模型":
            display_schelling_experiment()
        elif section == "3.4 技术扩散模型":
            display_technology_diffusion_experiment()
        elif section == "4.1 银行风险传染模型":
            display_bank_contagion_experiment()
        elif section == "4.2 信贷市场定价模型":
            display_credit_pricing_experiment()
        elif section == "4.3 激励机制设计模型":
            display_incentive_experiment()
        elif section == "5.1 农户作物选择模型":
            display_crop_choice_experiment()
        elif section == "5.2 农业保险采纳模型":
            display_insurance_adoption_experiment()
        elif section == "5.3 农村信贷风险模型":
            display_rural_credit_experiment()
        elif section == "5.4 粮食市场综合模型":
            display_grain_market_experiment()
    
    else:
        st.info(f"📝 {section} 的内容正在开发中...")
    
    # 底部导航按钮
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ 上一节", use_container_width=True):
            navigate_section(-1)
    
    with col2:
        if st.button("✅ 标记为已完成", use_container_width=True, type="primary"):
            st.session_state.progress[section] = True
            st.success("已标记为完成!")
            time.sleep(0.5)
            st.rerun()
    
    with col3:
        if st.button("下一节 ➡️", use_container_width=True):
            navigate_section(1)

def display_simple_experiment():
    """显示简单的交互实验"""
    
    st.markdown("---")
    st.markdown("<div class='experiment-card'>", unsafe_allow_html=True)
    st.markdown("### 🧪 交互实验: 简单ABM演示")
    
    st.markdown("""
    <div class='practice-box'>
    <b>实验目的:</b> 通过一个简单的模型,理解智能体、环境和交互的基本概念。
    
    <b>实验设置:</b> 创建一群智能体,每个智能体有初始财富和合作倾向。
    合作者财富稳定增长,非合作者财富波动较大。
    </div>
    """, unsafe_allow_html=True)
    
    # 参数设置
    col1, col2 = st.columns(2)
    with col1:
        n_agents = st.slider("智能体数量", 10, 200, 50, 10)
        n_steps = st.slider("模拟步数", 10, 100, 50, 10)
    
    with col2:
        st.markdown("**实验说明:**")
        st.markdown("- 🟢 合作者: 稳定收益 0~5")
        st.markdown("- 🔴 非合作者: 波动收益 -2~8")
        st.markdown("- 观察财富分布如何演化")
    
    # 运行模拟
    if st.button("▶️ 运行实验", type="primary", use_container_width=True):
        with st.spinner("模拟运行中..."):
            model = SimpleModel(n_agents=n_agents)
            
            # 运行模拟
            for _ in range(n_steps):
                model.step()
            
            # 获取数据
            results = model.datacollector.get_model_vars_dataframe()
            
            # 可视化
            col1, col2 = st.columns(2)
            
            with col1:
                # 财富演化
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=results.index,
                    y=results['平均财富'],
                    mode='lines+markers',
                    name='平均财富',
                    line=dict(color='#2196F3', width=2)
                ))
                fig1.update_layout(
                    title="平均财富演化",
                    xaxis_title="时间步",
                    yaxis_title="平均财富",
                    height=400
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # 合作比例
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=results.index,
                    y=results['合作比例'] * 100,
                    mode='lines+markers',
                    name='合作比例',
                    line=dict(color='#4CAF50', width=2),
                    fill='tozeroy'
                ))
                fig2.update_layout(
                    title="合作者比例",
                    xaxis_title="时间步",
                    yaxis_title="合作比例 (%)",
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # 最终分布
            final_wealth = [a.wealth for a in model.agents]
            cooperators = [a.wealth for a in model.agents if a.cooperate]
            non_cooperators = [a.wealth for a in model.agents if not a.cooperate]
            
            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(
                x=cooperators,
                name='合作者',
                opacity=0.7,
                marker_color='#4CAF50'
            ))
            fig3.add_trace(go.Histogram(
                x=non_cooperators,
                name='非合作者',
                opacity=0.7,
                marker_color='#F44336'
            ))
            fig3.update_layout(
                title="最终财富分布",
                xaxis_title="财富",
                yaxis_title="频数",
                barmode='overlay',
                height=400
            )
            st.plotly_chart(fig3, use_container_width=True)
            
            # 结果解读
            st.markdown("""
            <div class='theory-box'>
            <b>📊 结果解读:</b><br>
            - 观察合作者和非合作者的财富演化差异<br>
            - 理解<b>异质性</b>: 不同类型的智能体具有不同的行为规则<br>
            - 理解<b>涌现性</b>: 整体的财富分布模式是从个体行为中涌现出来的<br>
            - 尝试调整参数,观察系统行为如何变化
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


def display_forest_fire_experiment():
    """森林火灾传播实验"""
    st.markdown("---")
    st.markdown("<div class='experiment-card'>", unsafe_allow_html=True)
    st.markdown("### 🧪 交互实验: 森林火灾传播")
    
    st.markdown("""
    <div class='experiment-note'>
    <b>实验目标:</b> 观察森林密度对火灾传播范围的影响,发现<b>相变现象</b>。
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 参数设置")
        density = st.slider("森林密度", 0.0, 1.0, 0.6, 0.05, 
                           help="树木占据网格的比例")
        grid_size = st.slider("网格大小", 20, 80, 50, 10)
        
        run_fire = st.button("🔥 开始火灾模拟", type="primary", use_container_width=True)
    
    with col2:
        if run_fire:
            model = ForestFireModel(width=grid_size, height=grid_size, density=density)
            
            # 创建容器显示动画
            chart_placeholder = st.empty()
            
            # 运行模拟
            steps = 0
            while np.sum(model.grid == 2) > 0 and steps < 200:
                model.step()
                steps += 1
                
                # 每5步更新一次显示
                if steps % 5 == 0:
                    with chart_placeholder.container():
                        # 展示网格
                        fig, ax = plt.subplots(figsize=(6, 6))
                        colors = ['white', 'green', 'red']
                        cmap = plt.matplotlib.colors.ListedColormap(colors)
                        ax.imshow(model.grid, cmap=cmap, vmin=0, vmax=2)
                        ax.set_title(f"步数: {steps}")
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close()
            
            # 显示结果
            df = model.datacollector.get_model_vars_dataframe()
            final_burned = df["Burned"].iloc[-1]
            total_cells = grid_size * grid_size
            burned_pct = final_burned / total_cells
            
            st.markdown(f"""
            <div class='result-box'>
            <b>📊 实验结果:</b><br>
            - 森林密度: {density:.2f}<br>
            - 燃烧区域比例: {burned_pct:.2%}<br>
            - 火灾是否传播到右侧: <b>{'YES' if model.grid[:, -1].max() == 0 else 'NO'}</b><br>
            <br>
            <b>💡 提示:</b> 尝试调整密度到<b>0.59</b>附近,观察临界现象!
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


def display_sir_experiment():
    """病毒传播实验"""
    st.markdown("---")
    st.markdown("<div class='experiment-card'>", unsafe_allow_html=True)
    st.markdown("### 🦠 交互实验: 病毒传播 (SIR模型)")
    
    st.markdown("""
    <div class='experiment-note'>
    <b>实验目标:</b> 理解<b>基本传染数R0</b>对疫情爆发的影响。
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 参数设置")
        beta = st.slider("传染率 β", 0.0, 1.0, 0.3, 0.05,
                        help="单次接触被感染的概率")
        gamma = st.slider("康复率 γ", 0.0, 0.5, 0.1, 0.01,
                         help="每步康复的概率")
        
        r0 = (beta / gamma) * 8 if gamma > 0 else 0  # 简化计算(假设8个邻居)
        st.metric("基本传染数 R₀", f"{r0:.2f}")
        
        if r0 > 1:
            st.warning("⚠️ R₀ > 1: 疫情将爆发")
        else:
            st.success("✅ R₀ ≤ 1: 疫情将消退")
        
        run_sir = st.button("🦠 开始疫情模拟", type="primary", use_container_width=True)
    
    with col2:
        if run_sir:
            model = SIRModel(width=30, height=30, density=0.9, 
                           beta=beta, gamma=gamma, initial_infected=5)
            
            # 运行100步
            for _ in range(100):
                model.step()
                if all(agent.state != "I" for agent in model.agents):
                    break
            
            # 绘制SIR曲线
            df = model.datacollector.get_model_vars_dataframe()
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(df["Susceptible"], label="易感者 S", color='blue', linewidth=2)
            ax.plot(df["Infected"], label="感染者 I", color='red', linewidth=2)
            ax.plot(df["Recovered"], label="康复者 R", color='green', linewidth=2)
            ax.set_xlabel("时间步")
            ax.set_ylabel("人数")
            ax.set_title("SIR疫情演化")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            # 统计结果
            peak_infected = df["Infected"].max()
            final_susceptible = df["Susceptible"].iloc[-1]
            attack_rate = 1 - final_susceptible / df["Susceptible"].iloc[0]
            
            st.markdown(f"""
            <div class='result-box'>
            <b>📊 实验结果:</b><br>
            - R₀ = {r0:.2f}<br>
            - 感染峰值: {int(peak_infected)} 人<br>
            - 攻击率: {attack_rate:.2%} (最终被感染的比例)<br>
            <br>
            <b>💡 提示:</b> 调整β和γ使R₀接近1,观察临界点附近的不确定性!
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


def display_schelling_experiment():
    """Schelling隔离实验"""
    st.markdown("---")
    st.markdown("<div class='experiment-card'>", unsafe_allow_html=True)
    st.markdown("### 🏡 交互实验: Schelling居住隔离")
    
    st.markdown("""
    <div class='experiment-note'>
    <b>实验目标:</b> 观察<b>微观偏好</b>如何导致<b>宏观隔离</b>。
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 参数设置")
        homophily = st.slider("满意阈值", 0.0, 1.0, 0.3, 0.05,
                            help="最低同类邻居比例要求")
        
        st.info(f"💭 意义: 至少 {homophily:.0%} 的邻居是同类才满意")
        
        run_schelling = st.button("🏡 开始隔离模拟", type="primary", use_container_width=True)
    
    with col2:
        if run_schelling:
            model = SchellingModel(width=40, height=40, density=0.9, 
                                  homophily=homophily, minority_pc=0.5)
            
            # 运行100步
            initial_seg = model.calculate_segregation()
            for _ in range(100):
                model.step()
                unhappy_count = sum(1 for a in model.agents if not a.happy)
                if unhappy_count == 0:
                    break
            
            # 绘制网格
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
            # 初始状态(重新生成)
            init_model = SchellingModel(width=40, height=40, density=0.9, 
                                       homophily=homophily, minority_pc=0.5)
            grid1 = np.full((40, 40), -1.0)
            for agent in init_model.agents:
                x, y = agent.pos
                grid1[x][y] = agent.type
            
            ax1.imshow(grid1, cmap='RdBu', vmin=-1, vmax=1)
            ax1.set_title("初始状态")
            ax1.axis('off')
            
            # 最终状态
            grid2 = np.full((40, 40), -1.0)
            for agent in model.agents:
                x, y = agent.pos
                grid2[x][y] = agent.type
            
            ax2.imshow(grid2, cmap='RdBu', vmin=-1, vmax=1)
            ax2.set_title("最终状态")
            ax2.axis('off')
            
            st.pyplot(fig)
            plt.close()
            
            # 绘制隔离度曲线
            df = model.datacollector.get_model_vars_dataframe()
            fig2, ax = plt.subplots(figsize=(8, 4))
            ax.plot(df["Segregation"], color='purple', linewidth=2)
            ax.axhline(y=homophily, color='red', linestyle='--', label=f'设定阈值: {homophily:.2f}')
            ax.set_xlabel("时间步")
            ax.set_ylabel("隔离度")
            ax.set_title("隔离度演化")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig2)
            plt.close()
            
            # 统计结果
            final_seg = model.calculate_segregation()
            happy_pct = sum(1 for a in model.agents if a.happy) / len(model.agents)
            
            st.markdown(f"""
            <div class='result-box'>
            <b>📊 实验结果:</b><br>
            - 个人阈值: {homophily:.2f} (只要{homophily:.0%}同类就满意)<br>
            - 初始隔离度: {initial_seg:.2f}<br>
            - 最终隔离度: <b>{final_seg:.2f}</b> (实际隔离远高于阈值!)<br>
            - 满意比例: {happy_pct:.2%}<br>
            <br>
            <b>💡 关键发现:</b> 即使每个人只有<b>微弱的偏好</b>,最终也会形成<b>强烈的隔离</b>!
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


def display_technology_diffusion_experiment():
    """技术扩散S型曲线实验"""
    st.markdown("---")
    st.markdown("<div class='experiment-card'>", unsafe_allow_html=True)
    st.markdown("### 📈 交互实验: 技术扩散S型曲线")
    st.markdown("""
    <div class='experiment-note'>
    <b>实验目标:</b> 观察在社会网络中,技术采纳率如何随时间呈现S型扩散。
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 参数设置")
        n_agents = st.slider("主体数量", 50, 400, 200, 50)
        social_weight = st.slider("社会影响权重", 0.0, 1.0, 0.5, 0.1)
        max_steps = st.slider("模拟步数", 20, 200, 80, 10)
        run = st.button("📈 开始扩散模拟", type="primary", use_container_width=True)
    
    with col2:
        if run:
            # 简化: 用均匀随机图近似社会网络
            adjacency = np.random.rand(n_agents, n_agents) < 0.02
            np.fill_diagonal(adjacency, False)
            
            innovativeness = np.random.beta(2, 5, size=n_agents)
            adopted = np.zeros(n_agents, dtype=bool)
            adopters_over_time = []
            
            # 初始一小部分创新者
            initial_innovators = np.random.choice(n_agents, size=max(1, n_agents // 20), replace=False)
            adopted[initial_innovators] = True
            
            for t in range(max_steps):
                adopters_over_time.append(adopted.mean())
                new_adopted = adopted.copy()
                
                for i in range(n_agents):
                    if adopted[i]:
                        continue
                    neighbors = np.where(adjacency[i])[0]
                    if len(neighbors) == 0:
                        social_influence = 0
                    else:
                        social_influence = adopted[neighbors].mean() * social_weight
                    p = 0.3 * innovativeness[i] + 0.7 * social_influence
                    p = np.clip(p, 0, 1)
                    if np.random.rand() < p:
                        new_adopted[i] = True
                adopted = new_adopted
                if adopted.all():
                    adopters_over_time.append(1.0)
                    break
            
            adopters_array = np.array(adopters_over_time)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(adopters_array * 100, marker='o')
            ax.set_xlabel("时间步")
            ax.set_ylabel("采纳率 (%)")
            ax.set_title("技术扩散S型曲线")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            st.markdown(f"""
            <div class='result-box'>
            <b>📊 实验结果:</b><br>
            - 最终采纳率: {adopters_array[-1]*100:.1f}%<br>
            - 达到50%采纳所需时间: {np.argmax(adopters_array>=0.5) if (adopters_array>=0.5).any() else '未达到'} 步<br>
            <br>
            <b>💡 提示:</b> 调高社会影响权重可以显著加快扩散速度,并使曲线更“陡峭”。
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


def navigate_section(direction: int):
    """导航到上一节或下一节"""
    # 获取所有章节列表
    all_sections = []
    for chapter_content in LEARNING_PATH.values():
        all_sections.extend(chapter_content['sections'])
    
    # 找到当前位置
    current_idx = all_sections.index(st.session_state.current_section)
    new_idx = current_idx + direction
    
    # 边界检查
    if 0 <= new_idx < len(all_sections):
        st.session_state.current_section = all_sections[new_idx]
        
        # 更新当前章节
        for chapter, content in LEARNING_PATH.items():
            if st.session_state.current_section in content['sections']:
                st.session_state.current_chapter = chapter
                break
        
        st.rerun()


def display_bank_contagion_experiment():
    """银行风险传染实验"""
    st.markdown("---")
    st.markdown("<div class='experiment-card'>", unsafe_allow_html=True)
    st.markdown("### 🏦 交互实验: 银行网络风险传染")
    st.markdown("""
    <div class='experiment-note'>
    <b>实验目标:</b> 观察网络结构如何影响银行间风险传染速度。
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 参数设置")
        n_banks = st.slider("银行数量", 10, 50, 20, 5)
        network_type = st.selectbox("网络类型", ["随机网络", "无标度网络"])
        initial_shock = st.slider("初始冲击银行数", 1, 5, 1, 1)
        run = st.button("🏦 开始传染模拟", type="primary", use_container_width=True)
    
    with col2:
        if run:
            import networkx as nx
            if network_type == "随机网络":
                G = nx.erdos_renyi_graph(n_banks, 0.15)
            else:
                G = nx.barabasi_albert_graph(n_banks, 2)
            
            capital = np.random.uniform(50, 150, size=n_banks)
            exposure = nx.to_numpy_array(G) * np.random.uniform(5, 20, size=(n_banks, n_banks))
            np.fill_diagonal(exposure, 0)
            
            defaulted = np.zeros(n_banks, dtype=bool)
            initial_defaulted = np.random.choice(n_banks, size=initial_shock, replace=False)
            defaulted[initial_defaulted] = True
            
            default_history = [defaulted.sum()]
            for t in range(10):
                new_defaulted = defaulted.copy()
                for i in range(n_banks):
                    if defaulted[i]:
                        continue
                    loss = exposure[i, defaulted].sum()
                    if loss > capital[i]:
                        new_defaulted[i] = True
                defaulted = new_defaulted
                default_history.append(defaulted.sum())
                if defaulted.all():
                    break
            
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(default_history, marker='o', color='red', linewidth=2)
            ax.set_xlabel("传染轮次")
            ax.set_ylabel("违约银行数")
            ax.set_title("银行违约传染过程")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            final_default_rate = defaulted.sum() / n_banks
            contagion_rate = (defaulted.sum() - initial_shock) / initial_shock if initial_shock > 0 else 0
            
            st.markdown(f"""
            <div class='result-box'>
            <b>📊 实验结果:</b><br>
            - 网络类型: {network_type}<br>
            - 最终违约率: {final_default_rate:.1%}<br>
            - 网络传染率: {contagion_rate:.1f}x（每个初始冲击平均引发额外违约数）<br>
            <br>
            <b>💡 提示:</b> 无标度网络通常更脆弱,因为枢纽节点集中风险。
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


def display_credit_pricing_experiment():
    """信贷定价逆向选择实验"""
    st.markdown("---")
    st.markdown("<div class='experiment-card'>", unsafe_allow_html=True)
    st.markdown("### 💰 交互实验: 信贷市场逆向选择")
    st.markdown("""
    <div class='experiment-note'>
    <b>实验目标:</b> 观察利率如何影响借款人池质量和银行收益。
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 参数设置")
        max_rate = st.slider("最高利率", 0.05, 0.30, 0.15, 0.01)
        n_borrowers = 300
        run = st.button("💰 运行利率扫描", type="primary", use_container_width=True)
    
    with col2:
        if run:
            risk_types = np.random.choice([0.05, 0.15, 0.30], size=n_borrowers, p=[0.3, 0.4, 0.3])
            expected_returns = np.where(risk_types == 0.05, 0.08,
                                       np.where(risk_types == 0.15, 0.12, 0.18))
            
            rates = np.linspace(0.02, max_rate, 50)
            bank_profits = []
            avg_defaults = []
            
            for rate in rates:
                applicants_mask = expected_returns > rate
                if not applicants_mask.any():
                    bank_profits.append(0)
                    avg_defaults.append(0)
                    continue
                applicants_risk = risk_types[applicants_mask]
                avg_default = applicants_risk.mean()
                profit = (1 - avg_default) * rate - avg_default
                bank_profits.append(profit)
                avg_defaults.append(avg_default)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            ax1.plot(rates * 100, bank_profits, color='blue', linewidth=2)
            ax1.set_xlabel("利率 (%)")
            ax1.set_ylabel("银行期望收益")
            ax1.set_title("银行收益 vs 利率")
            ax1.grid(True, alpha=0.3)
            optimal_idx = np.argmax(bank_profits)
            ax1.axvline(rates[optimal_idx] * 100, color='red', linestyle='--', label=f'最优利率: {rates[optimal_idx]*100:.1f}%')
            ax1.legend()
            
            ax2.plot(rates * 100, np.array(avg_defaults) * 100, color='orange', linewidth=2)
            ax2.set_xlabel("利率 (%)")
            ax2.set_ylabel("平均违约率 (%)")
            ax2.set_title("借款人池违约率 vs 利率")
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            st.markdown(f"""
            <div class='result-box'>
            <b>📊 实验结果:</b><br>
            - 最优利率: {rates[optimal_idx]*100:.2f}%<br>
            - 最大银行收益: {bank_profits[optimal_idx]:.4f}<br>
            <br>
            <b>💡 关键发现:</b> 利率并非越高越好,过高利率会驱逐优质借款人,导致平均风险上升。
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


def display_incentive_experiment():
    """激励机制实验"""
    st.markdown("---")
    st.markdown("<div class='experiment-card'>", unsafe_allow_html=True)
    st.markdown("### 🎯 交互实验: 动态激励机制")
    st.markdown("""
    <div class='experiment-note'>
    <b>实验目标:</b> 观察奖金系数如何影响代理人努力程度和绩效。
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 参数设置")
        bonus_rate = st.slider("奖金系数", 0.0, 1.0, 0.5, 0.1)
        n_periods = st.slider("模拟期数", 10, 100, 30, 5)
        run = st.button("🎯 运行激励模拟", type="primary", use_container_width=True)
    
    with col2:
        if run:
            efforts = []
            performances = []
            
            for t in range(n_periods):
                effort_cost = lambda e: e ** 2 / 2
                utility = lambda e: bonus_rate * 100 * e - effort_cost(e)
                e_optimal = bonus_rate * 100
                e_optimal = np.clip(e_optimal, 0, 10)
                efforts.append(e_optimal)
                
                performance = e_optimal * 100 + np.random.normal(0, 20)
                performance = max(0, performance)
                performances.append(performance)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            ax1.plot(efforts, color='green', linewidth=2)
            ax1.set_xlabel("期数")
            ax1.set_ylabel("代理人努力程度")
            ax1.set_title("努力程度演化")
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(performances, color='blue', linewidth=2)
            ax2.set_xlabel("期数")
            ax2.set_ylabel("绩效输出")
            ax2.set_title("绩效演化（努力+运气）")
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            avg_effort = np.mean(efforts)
            avg_perf = np.mean(performances)
            
            st.markdown(f"""
            <div class='result-box'>
            <b>📊 实验结果:</b><br>
            - 奖金系数: {bonus_rate:.2f}<br>
            - 平均努力程度: {avg_effort:.2f}<br>
            - 平均绩效: {avg_perf:.1f}<br>
            <br>
            <b>💡 提示:</b> 提高奖金系数可增加努力,但需平衡风险转移成本。
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


def display_crop_choice_experiment():
    """农户作物选择实验"""
    st.markdown("---")
    st.markdown("<div class='experiment-card'>", unsafe_allow_html=True)
    st.markdown("### 🌾 交互实验: 农户作物选择")
    st.markdown("""
    <div class='experiment-note'>
    <b>实验目标:</b> 观察价格波动与邻里学习如何影响作物种植结构。
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 参数设置")
        corn_price = st.slider("玉米价格", 1.0, 3.0, 2.0, 0.1)
        soybean_price = st.slider("大豆价格", 1.0, 3.0, 1.8, 0.1)
        social_learning = st.slider("社会学习权重", 0.0, 1.0, 0.3, 0.1)
        n_farmers = 100
        n_steps = 20
        run = st.button("🌾 开始作物选择模拟", type="primary", use_container_width=True)
    
    with col2:
        if run:
            corn_yield = 5.0
            soybean_yield = 3.0
            corn_cost = 4.0
            soybean_cost = 2.5
            
            corn_profit = corn_price * corn_yield - corn_cost
            soybean_profit = soybean_price * soybean_yield - soybean_cost
            
            crop_choices = np.random.choice([0, 1], size=n_farmers)
            corn_ratio_over_time = []
            
            for t in range(n_steps):
                # 0 表示玉米, 1 表示大豆, 因此玉米种植比例 = 1 - 平均值
                corn_ratio_over_time.append(1 - crop_choices.mean())
                new_choices = crop_choices.copy()
                
                for i in range(n_farmers):
                    if np.random.rand() < social_learning:
                        neighbors_idx = np.random.choice(n_farmers, size=5, replace=False)
                        popular = np.bincount(crop_choices[neighbors_idx]).argmax()
                        new_choices[i] = popular
                    else:
                        if corn_profit > soybean_profit:
                            new_choices[i] = 0
                        else:
                            new_choices[i] = 1
                crop_choices = new_choices
            
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(np.array(corn_ratio_over_time) * 100, label="玉米种植比例", color='gold', linewidth=2)
            ax.plot((1 - np.array(corn_ratio_over_time)) * 100, label="大豆种植比例", color='green', linewidth=2)
            ax.set_xlabel("时间步")
            ax.set_ylabel("种植比例 (%)")
            ax.set_title("作物种植结构演化")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            final_corn_ratio = corn_ratio_over_time[-1]
            
            st.markdown(f"""
            <div class='result-box'>
            <b>📊 实验结果:</b><br>
            - 玉米收益: {corn_profit:.2f} | 大豆收益: {soybean_profit:.2f}<br>
            - 最终玉米种植比例: {final_corn_ratio*100:.1f}%<br>
            <br>
            <b>💡 提示:</b> 社会学习权重越高,价格信号传导越慢,种植结构调整有滞后性。
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


def display_insurance_adoption_experiment():
    """农业保险采纳实验"""
    st.markdown("---")
    st.markdown("<div class='experiment-card'>", unsafe_allow_html=True)
    st.markdown("### 🛡️ 交互实验: 农业保险采纳")
    st.markdown("""
    <div class='experiment-note'>
    <b>实验目标:</b> 观察政府补贴如何影响农户的投保决策。
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 参数设置")
        subsidy_rate = st.slider("补贴比例", 0.0, 0.9, 0.6, 0.1)
        run = st.button("🛡️ 运行投保分析", type="primary", use_container_width=True)
    
    with col2:
        if run:
            subsidy_levels = np.arange(0, 1, 0.05)
            adoption_rates = []
            
            for sub in subsidy_levels:
                premium_full = 100
                premium_farmer = premium_full * (1 - sub)
                wealth = 1000
                loss = 300
                prob = 0.1
                
                risk_aversion = 0.5
                u = lambda w: (w ** (1 - risk_aversion)) / (1 - risk_aversion)
                
                eu_no_ins = (1 - prob) * u(wealth) + prob * u(wealth - loss)
                eu_with_ins = (1 - prob) * u(wealth - premium_farmer) + prob * u(wealth - premium_farmer)
                
                adoption = 1 if eu_with_ins > eu_no_ins else 0
                adoption_rates.append(adoption)
            
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(subsidy_levels * 100, np.array(adoption_rates) * 100, marker='o', linewidth=2, color='blue')
            ax.axvline(subsidy_rate * 100, color='red', linestyle='--', label=f'当前补贴: {subsidy_rate*100:.0f}%')
            ax.set_xlabel("补贴比例 (%)")
            ax.set_ylabel("投保率 (%)")
            ax.set_title("投保率 vs 补贴比例")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            threshold_idx = np.argmax(adoption_rates)
            threshold_subsidy = subsidy_levels[threshold_idx]
            
            st.markdown(f"""
            <div class='result-box'>
            <b>📊 实验结果:</b><br>
            - 当前补贴比例: {subsidy_rate*100:.0f}%<br>
            - 自愿投保所需最低补贴: {threshold_subsidy*100:.0f}%<br>
            <br>
            <b>💡 关键发现:</b> 补贴降低农户保费负担,可显著提高投保积极性。
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


def display_rural_credit_experiment():
    """农村信贷风险实验"""
    st.markdown("---")
    st.markdown("<div class='experiment-card'>", unsafe_allow_html=True)
    st.markdown("### 🏦 交互实验: 农村信贷风险")
    st.markdown("""
    <div class='experiment-note'>
    <b>实验目标:</b> 观察信用评级如何影响借贷利率和违约风险。
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 参数设置")
        insurance_coverage = st.slider("保险覆盖率", 0.0, 1.0, 0.0, 0.1)
        n_farmers = 100
        run = st.button("🏦 运行信贷分析", type="primary", use_container_width=True)
    
    with col2:
        if run:
            wealth = np.random.lognormal(10, 1, size=n_farmers)
            land = np.random.uniform(5, 50, size=n_farmers)
            education = np.random.choice([0, 0.5, 1], size=n_farmers, p=[0.4, 0.4, 0.2])
            credit_history = np.random.uniform(0, 1, size=n_farmers)
            
            credit_scores = (wealth / 10000 * 0.3 + land / 50 * 0.2 + 
                           education * 0.2 + credit_history * 0.3)
            
            base_default_prob = np.where(credit_scores > 0.7, 0.05,
                                        np.where(credit_scores > 0.5, 0.15, 0.30))
            
            has_insurance = np.random.rand(n_farmers) < insurance_coverage
            actual_default_prob = np.where(has_insurance, base_default_prob * 0.5, base_default_prob)
            
            interest_rates = np.where(credit_scores > 0.7, 0.05,
                                     np.where(credit_scores > 0.5, 0.08, 0.12))
            interest_rates = np.where(has_insurance, interest_rates - 0.01, interest_rates)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            ax1.scatter(credit_scores, interest_rates * 100, c=has_insurance, cmap='RdYlGn', alpha=0.6)
            ax1.set_xlabel("信用评分")
            ax1.set_ylabel("利率 (%)")
            ax1.set_title("信用评分 vs 利率")
            ax1.grid(True, alpha=0.3)
            
            ax2.scatter(credit_scores, actual_default_prob * 100, c=has_insurance, cmap='RdYlGn', alpha=0.6)
            ax2.set_xlabel("信用评分")
            ax2.set_ylabel("违约概率 (%)")
            ax2.set_title("信用评分 vs 违约风险")
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            avg_default_no_ins = base_default_prob[~has_insurance].mean() if (~has_insurance).any() else 0
            avg_default_with_ins = actual_default_prob[has_insurance].mean() if has_insurance.any() else 0
            
            st.markdown(f"""
            <div class='result-box'>
            <b>📊 实验结果:</b><br>
            - 保险覆盖率: {insurance_coverage*100:.0f}%<br>
            - 无保险者平均违约率: {avg_default_no_ins*100:.1f}%<br>
            - 有保险者平均违约率: {avg_default_with_ins*100:.1f}%<br>
            <br>
            <b>💡 银保联动:</b> 保险降低违约风险,银行可给予更优惠利率。
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


def display_grain_market_experiment():
    """粮食市场综合模型实验"""
    st.markdown("---")
    st.markdown("<div class='experiment-card'>", unsafe_allow_html=True)
    st.markdown("### 🌾 交互实验: 粮食市场政策组合")
    st.markdown("""
    <div class='experiment-note'>
    <b>实验目标:</b> 观察关税+补贴+托市价政策对粮食自给率的影响。
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 参数设置")
        tariff_rate = st.slider("关税税率", 0.0, 0.5, 0.2, 0.05)
        floor_price = st.slider("托市价(元/斤)", 1.0, 3.0, 2.0, 0.1)
        subsidy_rate = st.slider("保险补贴率", 0.0, 0.8, 0.6, 0.1)
        run = st.button("🌾 运行粮食市场模拟", type="primary", use_container_width=True)
    
    with col2:
        if run:
            world_price = 1.8
            import_price = world_price * (1 + tariff_rate)
            
            n_farmers = 200
            yields = np.random.normal(500, 50, size=n_farmers)
            costs = np.random.uniform(1.2, 1.6, size=n_farmers)
            
            has_insurance = np.random.rand(n_farmers) < (0.3 + subsidy_rate * 0.5)
            
            loss_prob = 0.1
            actual_yields = np.where(has_insurance, yields, 
                                    np.where(np.random.rand(n_farmers) < loss_prob, yields * 0.5, yields))
            
            total_supply = actual_yields.sum()
            population = 50000
            demand = population * 0.5
            
            if total_supply < demand:
                import_quantity = demand - total_supply
                market_price = import_price
            else:
                import_quantity = 0
                market_price_base = demand / total_supply * 2.0
                market_price = max(floor_price, market_price_base)
            
            self_sufficiency = total_supply / demand
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            ax1.bar(['supply', 'demand'], [total_supply, demand], color=['green', 'orange'])
            ax1.set_ylabel("数量 (斤)")
            ax1.set_title("粮食供需对比")
            ax1.grid(True, axis='y', alpha=0.3)
            
            policy_labels = [f'世界价', f'进口价\n(含关税)', f'国内价', f'托市价']
            policy_prices = [world_price, import_price, market_price, floor_price]
            colors = ['gray', 'blue', 'green', 'red']
            ax2.bar(policy_labels, policy_prices, color=colors, alpha=0.7)
            ax2.set_ylabel("价格 (元/斤)")
            ax2.set_title("价格体系对比")
            ax2.grid(True, axis='y', alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            st.markdown(f"""
            <div class='result-box'>
            <b>📊 实验结果:</b><br>
            - 粮食自给率: {self_sufficiency*100:.1f}%<br>
            - 市场价格: {market_price:.2f} 元/斤<br>
            - 进口数量: {import_quantity:.0f} 斤<br>
            - 投保率: {(has_insurance.mean()*100):.1f}%<br>
            <br>
            <b>💡 政策启示:</b> 关税保护国内价格,补贴提高投保率,托市价稳定农户收入。
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


def navigate_section(direction: int):
    # 获取所有章节列表
    all_sections = []
    for chapter_content in LEARNING_PATH.values():
        all_sections.extend(chapter_content['sections'])
    
    # 找到当前位置
    current_idx = all_sections.index(st.session_state.current_section)
    new_idx = current_idx + direction
    
    # 边界检查
    if 0 <= new_idx < len(all_sections):
        st.session_state.current_section = all_sections[new_idx]
        
        # 更新当前章节
        for chapter, content in LEARNING_PATH.items():
            if st.session_state.current_section in content['sections']:
                st.session_state.current_chapter = chapter
                break
        
        st.rerun()


def save_progress():
    """保存学习进度到本地文件"""
    progress_file = "learning_progress.json"
    try:
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.progress, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"保存失败: {e}")
        return False


def generate_learning_report():
    """生成学习报告"""
    total_sections = sum(len(v['sections']) for v in LEARNING_PATH.values())
    completed_sections = sum(st.session_state.progress.values())
    progress_pct = completed_sections / total_sections if total_sections > 0 else 0
    
    report = f"""
# ABM教学平台学习报告

**生成日期:** {datetime.now().strftime('%Y年%m月%d日 %H:%M')}

---

## 📊 学习进度总览

- **总体完成率:** {progress_pct:.1%} ({completed_sections}/{total_sections} 节)
- **当前学习位置:** {st.session_state.current_chapter} - {st.session_state.current_section}

---

## 📚 各章节学习明细

"""
    
    for chapter, content in LEARNING_PATH.items():
        chapter_completed = sum(st.session_state.progress.get(s, False) for s in content['sections'])
        chapter_total = len(content['sections'])
        chapter_pct = chapter_completed / chapter_total if chapter_total > 0 else 0
        
        report += f"\n### {content['icon']} {chapter}\n\n"
        report += f"**完成率:** {chapter_pct:.1%} ({chapter_completed}/{chapter_total})\n\n"
        
        for section in content['sections']:
            status = "✅ 已完成" if st.session_state.progress.get(section, False) else "⭕ 未完成"
            report += f"- {status} {section}\n"
        
        report += "\n"
    
    report += f"""
---

## 🎯 学习建议

"""
    
    if progress_pct < 0.3:
        report += """
**建议优先级:**
1. 系统学习第一章的ABM基本概念
2. 完成第二章的建模流程学习
3. 动手实践第三章的经典案例
"""
    elif progress_pct < 0.7:
        report += """
**建议优先级:**
1. 巩固已学内容,复习关键概念
2. 继续完成第四章的金融ABM应用
3. 尝试调整实验参数,加深理解
"""
    else:
        report += """
**建议优先级:**
1. 完成剩余章节的学习
2. 综合应用所学知识,尝试构建自己的模型
3. 阅读相关论文,扩展知识面
"""
    
    report += f"""

---

## 📝 备注

本报告由ABM教学实验平台自动生成。

**平台版本:** v1.0
**报告格式:** Markdown
"""
    
    return report


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == "__main__":
    main()
