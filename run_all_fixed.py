#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import sys

# 按照指定顺序的文件列表
files = [

    "宏观经济/1.1中国10债Non-Trend_api.ipynb",
    "宏观经济/1.2中国10债Non-Trend.ipynb",
    "宏观经济/1.3中国10债Trend_api.ipynb",
    "宏观经济/1.4中国10债Trend.ipynb",
#    "宏观经济/2.1美国10债Non-trend_api.ipynb",
#    "宏观经济/2.2美国10债Non-trend.ipynb",
#    "宏观经济/2.3美国10债trend_api.ipynb",
#    "宏观经济/2.4美国10债trend.ipynb",
    "宏观经济/3.1美元指数拟合残差(10美债)_api.ipynb",
    "宏观经济/3.2美元指数拟合残差(10美债).ipynb",
    "宏观经济/3.3美元指数_api.ipynb",
    "宏观经济/3.4美元指数.ipynb",

    "宏观经济/3.5美元指数2_api.ipynb",
    "宏观经济/3.6美元指数2.ipynb",

    "宏观经济/4.1USDCNY即期汇率_api.ipynb",
    "宏观经济/4.2USDCNY即期汇率.ipynb",
#    "宏观经济/4.3人民币汇率收盘价Non-Trend_api.ipynb",
#    "宏观经济/4.3人民币汇率收盘价Non-Trend.ipynb",
#    "宏观经济/4.5人民币汇率收盘价Trend_api.ipynb",
#    "宏观经济/4.6人民币汇率收盘价Trend.ipynb",
    "宏观经济/5.1美国10年通胀预测Non-Trend_api.ipynb",
    "宏观经济/5.2美国10年通胀预测Non-Trend.ipynb",
    "宏观经济/5.3美国10年通胀预测Trend_api.ipynb",
    "宏观经济/5.4美国10年通胀预测Trend.ipynb",
    "宏观经济/6.1欧元-美元_api.ipynb",
    "宏观经济/6.2欧元-美元.ipynb",
#    "宏观经济/7.1美国GDP_api.ipynb",
#    "宏观经济/7.2美国GDP.ipynb"
    "宏观经济/8.1美国10债Non-Trend_api.ipynb",
    "宏观经济/8.2美国10债Non-Trend.ipynb",
    "宏观经济/8.3美国10债Trend_api.ipynb",
    "宏观经济/8.4美国10债Trend.ipynb",



    "wti模型3.0/1.1美国RBOB汽油裂解api.ipynb",
    "wti模型3.0/1.2美国RBOB汽油裂解.ipynb",
    "wti模型3.0/1.3美国取暖油裂解_api.ipynb",
    "wti模型3.0/1.4美国取暖油裂解.ipynb",
    "wti模型3.0/1.5PADD3炼厂压裂裂解_api.ipynb",
    "wti模型3.0/1.6PADD3炼厂压裂裂解.ipynb",    
    "wti模型3.0/2.3WTI_连1-连4月差-残差项2_api.ipynb",
    "wti模型3.0/2.4WTI_连1-连4月差-残差项2.ipynb",
    "wti模型3.0/3.3wti_连1-4_2_api.ipynb",
    "wti模型3.0/3.4wti_连1-4_2.ipynb",
    "wti模型3.0/4.1wti_残差项api.ipynb",
    "wti模型3.0/4.3wti_残差项.ipynb",
    "wti模型3.0/5.1wti_原油合约价格api.ipynb",
    "wti模型3.0/5.3wti_原油合约价格_final_日度.ipynb",
    "wti模型3.0/5.4wti_原油合约价格_final_月度.ipynb",
    "wti模型3.0/6.1Brent-WTI价差api.ipynb",
    "wti模型3.0/6.2Brent-WTI价差.ipynb",   
#    "wti模型3.0/7.1Brent-Dubai_api.ipynb",
#    "wti模型3.0/7.2Brent-Dubai.ipynb",
#    "wti模型3.0/8.1迪拜油_api.ipynb",
#    "wti模型3.0/8.2迪拜油.ipynb"    


#   "汽柴煤油2.0/1.1中国汽油表需api.ipynb",
#   "汽柴煤油2.0/1.2中国汽油表需.ipynb",
#    "汽柴煤油2.0/2.1中国汽油社会库存api.ipynb",
#    "汽柴煤油2.0/2.2中国汽油社会库存.ipynb",
#    "汽柴煤油2.0/3.1中国汽油主营销售库存api.ipynb",
#    "汽柴煤油2.0/3.2中国汽油主营销售库存.ipynb",
#    "汽柴煤油2.0/4.1汽油独立炼厂表需api.ipynb",
#    "汽柴煤油2.0/4.2汽油独立炼厂表需.ipynb",
#    "汽柴煤油2.0/5.1汽油独立炼厂库存api.ipynb",
#    "汽柴煤油2.0/5.2汽油独立炼厂库存.ipynb",
    "汽柴煤油2.0/5.3中国汽油需求(多因子)_api.ipynb",
    "汽柴煤油2.0/5.4中国汽油需求(多因子).ipynb",
#    "汽柴煤油2.0/6.1.1山东汽油裂解_残差api.ipynb",
#    "汽柴煤油2.0/6.1.2山东汽油裂解_残差.ipynb",
#    "汽柴煤油2.0/6.2.1山东汽油裂解api.ipynb",
#    "汽柴煤油2.0/6.2.2山东汽油裂解.ipynb",
#    "汽柴煤油2.0/7.1.1山东柴油裂解差Non-trend_api.ipynb",
#    "汽柴煤油2.0/7.1.2山东柴油裂解差Non-trend.ipynb",
#    "汽柴煤油2.0/7.1.3山东柴油裂解差Trend_api.ipynb",
#    "汽柴煤油2.0/7.1.4山东柴油裂解差Trend.ipynb",
#    "汽柴煤油2.0/7.3中国柴油需求Non-trend_api.ipynb",
#    "汽柴煤油2.0/7.3中国柴油需求Non-trend.ipynb",
#    "汽柴煤油2.0/7.3中国柴油需求Trend_api.ipynb",
#    "汽柴煤油2.0/7.3中国柴油需求Trend.ipynb",   
    "汽柴煤油2.0/7.4中国柴油需求(多因子)_api.ipynb",
    "汽柴煤油2.0/7.4中国柴油需求(多因子).ipynb",
#    "汽柴煤油2.0/8.1柴油主营销售库存_api.ipynb",
#    "汽柴煤油2.0/8.2柴油主营销售库存.ipynb",
#    "汽柴煤油2.0/9.1柴油社会库存_api.ipynb",
#    "汽柴煤油2.0/9.2柴油社会库存.ipynb",
#    "汽柴煤油2.0/10.1柴油独立炼厂表需_api.ipynb",
#    "汽柴煤油2.0/10.2柴油独立炼厂表需.ipynb",
#    "汽柴煤油2.0/11.1柴油独立炼厂产量_api.ipynb",   
#    "汽柴煤油2.0/11.2柴油独立炼厂产量.ipynb",
#    "汽柴煤油2.0/12.1柴油独立炼厂库存_api.ipynb",
#    "汽柴煤油2.0/12.2柴油独立炼厂库存.ipynb",
#    "汽柴煤油2.0/13.1柴油裂解差拟合残差-库存_api.ipynb",
#    "汽柴煤油2.0/13.2柴油裂解差拟合残差-库存.ipynb",
#    "汽柴煤油2.0/14.1柴油裂解差(拟合残差)_api.ipynb",
#    "汽柴煤油2.0/14.2柴油裂解差(拟合残差).ipynb",
    "汽柴煤油2.0/14.3山东柴油裂解差(多因子)_api.ipynb",
    "汽柴煤油2.0/14.4山东柴油裂解差(多因子).ipynb",
    "汽柴煤油2.0/14.5山东汽油裂解差(多因子)_api.ipynb",
    "汽柴煤油2.0/14.6山东汽油裂解差(多因子).ipynb",
#    "汽柴煤油2.0/15.1煤柴价差日度api.ipynb",
#    "汽柴煤油2.0/15.2煤柴价差日度.ipynb",
    "汽柴煤油2.0/15.3新加坡航空煤油裂解价差拟合残差_布伦特迪拜api.ipynb",
    "汽柴煤油2.0/15.4新加坡航空煤油裂解价差拟合残差_布伦特迪拜.ipynb",
    #"汽柴煤油2.0/15.5新加坡航空煤油裂解价差api.ipynb",
    #"汽柴煤油2.0/15.6新加坡航空煤油裂解价差.ipynb",
    "汽柴煤油2.0/15.7 新加坡航空煤油裂解价差api.ipynb",
    "汽柴煤油2.0/15.8 新加坡航空煤油裂解价差.ipynb",
#    "汽柴煤油2.0/16.1Brent-汽油-柴油-煤油api.ipynb",
    "汽柴煤油2.0/16.1Brent-汽油-柴油-煤油api2.ipynb",
#    "汽柴煤油2.0/16.2Brent-汽油-柴油-煤油_预测.ipynb",
#    "汽柴煤油2.0/16.2Brent-汽油-柴油-煤油_预测2.ipynb",
    "汽柴煤油2.0/16.2Brent-汽油-柴油-煤油_预测3.ipynb",
#    "汽柴煤油2.0/27.1RBOB汽油裂解价差_api.ipynb",
#    "汽柴煤油2.0/27.2RBOB汽油裂解价差.ipynb"


    '燃料油/1.1新加坡高硫燃料油380裂解价差_api.ipynb',
    '燃料油/1.2新加坡高硫燃料油380裂解价差.ipynb',
    '燃料油/2.1新加坡高低硫燃料油价差_api.ipynb',
    '燃料油/2.2新加坡高低硫燃料油价差.ipynb',
    '燃料油/3.1新加坡0.5%低硫燃料油裂解价差_api.ipynb',
    '燃料油/3.2新加坡0.5%低硫燃料油裂解价差.ipynb',



    "动力煤/动力煤价格_api.ipynb",
    "动力煤/动力煤价格.ipynb",

    "天然气/1.1天然气TTF连1价格Non-Trend_api.ipynb",
    "天然气/1.2天然气TTF连1价格Non-Trend.ipynb",
    "天然气/1.3天然气TTF连1价格Trend_api.ipynb", 
    "天然气/1.4天然气TTF连1价格Trend.ipynb",
    "天然气/2.1美国天然气HH连1合约Non-Trend_api.ipynb",
    "天然气/2.2美国天然气HH连1合约Non-Trend.ipynb",
    "天然气/2.3美国天然气HH连1合约_api.ipynb",
    "天然气/2.4美国天然气HH连1合约.ipynb",
    "天然气/3.1JKM-TTF_api.ipynb",
    "天然气/3.2JKM-TTF.ipynb",
    "天然气/3.3JKM_api.ipynb",
    "天然气/3.4JKM.ipynb",


    '黑色/玻璃/1.玻璃期货价格拟合残差-企业库存_api.ipynb',
    '黑色/玻璃/2.玻璃期货价格拟合残差-企业库存.ipynb',
    '黑色/玻璃/3.玻璃期货价格_api.ipynb',
    '黑色/玻璃/4.玻璃期货价格.ipynb',
    '黑色/玻璃/5.1纯碱开工率_api.ipynb',
    '黑色/玻璃/5.2纯碱开工率.ipynb',


    '黑色/铁矿/1.1日均铁水产量api.ipynb',
    '黑色/铁矿/1.2日均铁水产量_Xgboost.ipynb',
    '黑色/铁矿/1.3日均铁水产量_超季节性_api.ipynb',
    '黑色/铁矿/1.4日均铁水产量_超季节性.ipynb',
    '黑色/铁矿/2.1铁矿期货价格拟合残差api.ipynb',
    '黑色/铁矿/2.2铁矿期货价格拟合残差.ipynb',
    '黑色/铁矿/2.3铁矿期货价格_api.ipynb',
    '黑色/铁矿/2.4铁矿期货价格.ipynb',

    '铜/铜_api.ipynb',
    '铜/铜.ipynb',


    "化工/2.2 PX-WTI价差_api.ipynb",
    "化工/2.2 PX-WTI价差.ipynb",
    "化工/3.乙二醇加权利润残差-总库存_api.ipynb",
    "化工/3.乙二醇加权利润残差-总库存.ipynb",
    "化工/4.乙二醇加权利润_api.ipynb",
    "化工/4.乙二醇加权利润.ipynb",
    "化工/5.1PTA加工费_api.ipynb",
    "化工/5.2PTA加工费.ipynb",
    "化工/6.1PTA现货价格_api.ipynb",
    "化工/6.2PTA现货价格.ipynb",


    '铝/1.1氧化铝周度表需non-trend_api.ipynb',
    '铝/1.2氧化铝周度表需non-trend.ipynb',
    '铝/1.3氧化铝周度表需trend_api.ipynb',
    '铝/1.4氧化铝周度表需trend.ipynb',
    '铝/2.1氧化铝周度产量_api.ipynb',
    '铝/2.2氧化铝周度产量.ipynb',
    '铝/3.1氧化铝周度库存_api.ipynb',
    '铝/3.2氧化铝周度库存.ipynb',
    '铝/4.1沪铝期货价格_api.ipynb',
    '铝/4.2沪铝期货价格.ipynb',
    '铝/5.1氧化铝价格拟合残差_总库存_api.ipynb',
    '铝/5.2氧化铝价格拟合残差_总库存.ipynb',
    '铝/5.3氧化铝价格_api.ipynb',
    '铝/5.4氧化铝价格.ipynb',

    '焦煤/1.焦煤港口库存_api.ipynb',
    '焦煤/2.焦煤港口库存.ipynb',
    '焦煤/3.焦煤288口岸监管区总库存_api.ipynb',
    '焦煤/4.焦煤288口岸监管区总库存.ipynb',
    '焦煤/5.焦煤煤矿库存_api.ipynb',
    '焦煤/6.焦煤煤矿库存.ipynb',
    '焦煤/7.焦煤上游总库存_api.ipynb',
    '焦煤/8.焦煤上游总库存.ipynb',

    '焦煤/9.主焦煤价格_临汾拟合残差_焦煤煤矿_港口库存_api.ipynb',
    '焦煤/10.主焦煤价格_临汾拟合残差_焦煤煤矿_港口库存.ipynb',
    '焦煤/11.主焦煤价格-临汾_api.ipynb',
    '焦煤/12.主焦煤价格-临汾.ipynb',

    '焦炭/1.焦化厂利润_api.ipynb',
    '焦炭/2.焦化厂利润.ipynb',
    '焦炭/3.1焦炭现货基差_api.ipynb',
    '焦炭/3.2焦炭现货基差.ipynb',
    '焦炭/4.1焦炭港口价格_api.ipynb',
    '焦炭/4.2焦炭港口价格.ipynb',
    '焦炭/5.1焦炭港口仓单价_api.ipynb',
    '焦炭/5.2焦炭港口仓单价.ipynb',
    '焦炭/6.1焦炭期货价格_api.ipynb',
    '焦炭/6.2焦炭期货价格.ipynb',


    "wti模型3.0/7.1Brent-Dubai_api.ipynb",
    "wti模型3.0/7.2Brent-Dubai.ipynb",
    "wti模型3.0/8.1迪拜油_api.ipynb",
    "wti模型3.0/8.2迪拜油.ipynb" ,



    "汽柴煤油2.0/中石化航空煤油Non-Trend_F0.2_api.ipynb",
    "汽柴煤油2.0/中石化航空煤油Non-Trend_F0.2.ipynb",
    "汽柴煤油2.0/中石化航空煤油Trend_api.ipynb",
    "汽柴煤油2.0/中石化航空煤油Trend.ipynb",
    '汽柴煤油2.0/17.1新加坡92汽油裂解_api.ipynb',
    '汽柴煤油2.0/17.2新加坡92汽油裂解.ipynb',  
    '汽柴煤油2.0/18.1新加坡10ppm_api.ipynb',
    '汽柴煤油2.0/18.2新加坡10ppm.ipynb',  
    '汽柴煤油2.0/19.1汽油出口利润(华东-新加坡)_api.ipynb',
    '汽柴煤油2.0/19.2汽油出口利润(华东-新加坡).ipynb',
    '汽柴煤油2.0/20.1柴油出口利润(华东-新加坡)_api.ipynb',
    '汽柴煤油2.0/20.2柴油出口利润(华东-新加坡).ipynb',
    '汽柴煤油2.0/21.1中国汽油出口计划量_api.ipynb',
    '汽柴煤油2.0/21.2中国汽油出口计划量.ipynb',
    '汽柴煤油2.0/22.1FU连1_连2_api.ipynb',
    '汽柴煤油2.0/22.2FU连1_连2.ipynb',
    '汽柴煤油2.0/22.3LU连1_连2_api.ipynb',
    '汽柴煤油2.0/22.4LU连1_连2.ipynb',
    '汽柴煤油2.0/23.1原油加工量_api.ipynb',
    '汽柴煤油2.0/23.2原油加工量.ipynb',
    '汽柴煤油2.0/24.1LU-FU_api.ipynb',
    '汽柴煤油2.0/24.2LU-FU.ipynb',
    '汽柴煤油2.0/25.1FU-BU_api.ipynb',
    '汽柴煤油2.0/25.2FU-BU.ipynb',
    '汽柴煤油2.0/26.1LU-BU_api.ipynb',
    '汽柴煤油2.0/26.2LU-BU.ipynb',
    '汽柴煤油2.0/27.1RBOB汽油裂解价差(均价)_api.ipynb',
    '汽柴煤油2.0/27.2RBOB汽油裂解价差(均价).ipynb',
    '汽柴煤油2.0/28.1 SC期货指数-Brent原油期货价格_api.ipynb',
    '汽柴煤油2.0/28.2 SC期货指数-Brent原油期货价格.ipynb',
    '汽柴煤油2.0/29.1 SC原油连1-连3月差_api.ipynb',
    '汽柴煤油2.0/29.2 SC原油连1-连3月差.ipynb',



    '汽柴煤油2.0/30.1 FU-SC_api.ipynb',
    '汽柴煤油2.0/30.2 FU-SC.ipynb',
    '汽柴煤油2.0/31.1 LU-SC_api.ipynb',
    '汽柴煤油2.0/31.2 LU-SC.ipynb',
    '汽柴煤油2.0/32.1 BU_SC_api.ipynb',
    '汽柴煤油2.0/32.2 BU_SC.ipynb',
    '汽柴煤油2.0/33.1 TA_SC_api.ipynb',
    '汽柴煤油2.0/33.2 TA_SC.ipynb',
    '汽柴煤油2.0/34.1SC期货指数_api.ipynb',
    '汽柴煤油2.0/34.2SC期货指数.ipynb',
    '汽柴煤油2.0/35.1TA期货指数_api.ipynb',
    '汽柴煤油2.0/35.2TA期货指数.ipynb',
    '汽柴煤油2.0/36.1 纯苯-Brent价差拟合残差-亚洲PX负荷_api.ipynb',
    '汽柴煤油2.0/36.2 纯苯-Brent价差拟合残差-亚洲PX负荷.ipynb',
    '汽柴煤油2.0/37.1 纯苯-Brent价差_api.ipynb',
    '汽柴煤油2.0/37.2 纯苯-Brent价差.ipynb',
    '汽柴煤油2.0/37.3 纯苯_api.ipynb',
    '汽柴煤油2.0/37.4 纯苯.ipynb',
    '汽柴煤油2.0/38.1PX-SC_api.ipynb',
    '汽柴煤油2.0/38.2PX-SC.ipynb',
    '汽柴煤油2.0/39.1PX_api.ipynb',
    '汽柴煤油2.0/39.2PX.ipynb',
    '汽柴煤油2.0/40.1 EB-SC（期货指数）拟合残差-纯苯-Brent价差_api.ipynb',
    '汽柴煤油2.0/40.2 EB-SC（期货指数）拟合残差-纯苯-Brent价差.ipynb',
    '汽柴煤油2.0/41.1 EB-SC(期货指数)_api.ipynb',
    '汽柴煤油2.0/41.2 EB-SC(期货指数).ipynb',
    '汽柴煤油2.0/42.1EB_api.ipynb',
    '汽柴煤油2.0/42.2EB.ipynb',



    '聚丙烯(PP)/1.1油制PP盘面利润拟合残差-WTI原油期货价格_api.ipynb',
    '聚丙烯(PP)/1.2油制PP盘面利润拟合残差-WTI原油期货价格.ipynb',
    '聚丙烯(PP)/1.3油制PP期货价格_api.ipynb',
    '聚丙烯(PP)/1.4油制PP期货价格.ipynb',


    '沥青/1.1沥青市场价华东地区拟合残差-Brent原油期货价格_api.ipynb',
    '沥青/1.2沥青市场价华东地区拟合残差-Brent原油期货价格.ipynb',
    '沥青/1.3沥青市场价华东地区_api.ipynb',
    '沥青/1.4沥青市场价华东地区.ipynb',
    '沥青/2.1沥青月差_api.ipynb',
    '沥青/2.2沥青月差.ipynb',


    '螺纹/1.1螺纹盘面利润_api.ipynb',
    '螺纹/1.2螺纹盘面利润.ipynb',
    '螺纹/2.1螺纹盘面成本_api.ipynb',
    '螺纹/2.2螺纹盘面成本.ipynb',
    '螺纹/3.1螺纹期货价格_api.ipynb',
    '螺纹/3.2螺纹期货价格.ipynb',

    '汽柴煤油2.0/43.1石脑油-Brent拟合残差_Brent_api.ipynb',
    '汽柴煤油2.0/43.2石脑油-Brent拟合残差_Brent.ipynb', 
    '汽柴煤油2.0/43.3石脑油-Brent拟合残差_api.ipynb',
    '汽柴煤油2.0/43.4石脑油-Brent拟合残差.ipynb',
    '汽柴煤油2.0/43.5石脑油_api.ipynb',
    '汽柴煤油2.0/43.6石脑油.ipynb',

    '汽柴煤油2.0/44.1PP-SC拟合残差_SC_api.ipynb',
    '汽柴煤油2.0/44.2PP-SC拟合残差_SC.ipynb', 
    '汽柴煤油2.0/44.3PP-SC_api.ipynb',
    '汽柴煤油2.0/44.4PP-SC.ipynb',
    '汽柴煤油2.0/44.5PP_api.ipynb',
    '汽柴煤油2.0/44.6PP.ipynb',

    '汽柴煤油2.0/45.1PE-SC拟合残差_SC_api.ipynb',
    '汽柴煤油2.0/45.2PE-SC拟合残差_SC.ipynb', 
    '汽柴煤油2.0/45.3PE-SC_api.ipynb',
    '汽柴煤油2.0/45.4PE-SC.ipynb',
    '汽柴煤油2.0/45.5 PE_api.ipynb',
    '汽柴煤油2.0/45.6 PE.ipynb',

    '汽柴煤油2.0/46.1 EG-SC拟合残差_SC_api.ipynb',
    '汽柴煤油2.0/46.2 EG-SC拟合残差_SC.ipynb', 
    '汽柴煤油2.0/46.3 EG-SC_api.ipynb',
    '汽柴煤油2.0/46.4 EG-SC.ipynb',
    '汽柴煤油2.0/46.5 EG_api.ipynb',
    '汽柴煤油2.0/46.6 EG.ipynb',

    '汽柴煤油2.0/47.1 纯苯-EB_api.ipynb',
    '汽柴煤油2.0/47.2 纯苯-EB.ipynb',
    '汽柴煤油2.0/48.1 苯乙烯-纯苯价差_api.ipynb',
    '汽柴煤油2.0/48.2 苯乙烯-纯苯价差.ipynb',


    '汽柴煤油2.0/49.1 欧洲柴油利润拟合残差_10ppm_api.ipynb',
    '汽柴煤油2.0/49.2 欧洲柴油利润拟合残差_10ppm.ipynb',
    '汽柴煤油2.0/49.3 欧洲柴油利润_api.ipynb',
    '汽柴煤油2.0/49.4 欧洲柴油利润.ipynb',
      
    '汽柴煤油2.0/50.1山东丙烯主流价-SC指数_api.ipynb',
    '汽柴煤油2.0/50.2山东丙烯主流价-SC指数.ipynb',
    '汽柴煤油2.0/50.3山东丙烯主流价_api.ipynb',
    '汽柴煤油2.0/50.4山东丙烯主流价.ipynb',

    '汽柴煤油2.0/51.1PG-SC拟合残差_原油指数_api.ipynb',
    '汽柴煤油2.0/51.2PG-SC拟合残差_原油指数.ipynb',
    '汽柴煤油2.0/51.3结算价_LPG指数_api.ipynb',
    '汽柴煤油2.0/51.4结算价_LPG指数.ipynb',

    '汽柴煤油2.0/52.1 山东丙烯-LPG_api.ipynb',
    '汽柴煤油2.0/52.2 山东丙烯-LPG.ipynb',
    '汽柴煤油2.0/53.1 PP-山东丙烯_api.ipynb',
    '汽柴煤油2.0/53.2 PP-山东丙烯.ipynb',
    '汽柴煤油2.0/54.1 PP-LPG_api.ipynb',
    '汽柴煤油2.0/54.2 PP-LPG.ipynb',

    '汽柴煤油2.0/55.1 PTA-EG_api.ipynb',
    '汽柴煤油2.0/55.2 PTA-EG.ipynb',   

    '汽柴煤油2.0/56.1 PTA-PX_api.ipynb',
    '汽柴煤油2.0/56.2 PTA-PX.ipynb',


    "上传数据_日度数据_日期限制.py",
    "上传数据_列表页_数据获取.py",
    "上传数据_合并数据.py",

    
]



def run_file(file):
    print("正在执行文件:", file)
    # 根据文件类型构造不同的命令
    if file.endswith('.py'):
        cmd = ["python", file]
    elif file.endswith('.ipynb'):
        # 使用 jupyter-nbconvert 执行 notebook 文件，同步执行并直接写回原文件（避免产生新文件），使用 --inplace 参数
        cmd = ["jupyter-nbconvert", "--to", "notebook", "--execute", "--inplace", file]
    else:
        print("未知文件类型:", file)
        return False

    try:
        # 执行命令，capture_output=True 用于捕获输出信息，check=True 如有错误会抛出 CalledProcessError
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        # 捕获到错误后，打印错误信息但继续执行
        print("执行文件时出现错误:", file)
        print("错误输出:", e.stderr)
        return False

def main():
    error_files = []
    for file in files:
        success = run_file(file)
        if not success:
            error_files.append(file)
    
    print("所有文件执行完毕！")
    if error_files:
        print("\n执行失败的文件列表:")
        for file in error_files:
            print(file)

if __name__ == "__main__":
    main()