import pandas as pd

def generate_and_fill_excel(
    daily_df,
    target_name,        # 写入的"预测标的"显示名
    TARGET_COL,
    output_path='update.xlsx'
):
    try:
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            workbook = writer.book
            # —— 计算三个汇总值 —— 
            
            # ========= 列表页 =========
            ws_list = workbook.add_worksheet('列表页')
            writer.sheets['列表页'] = ws_list

            headers = ['预测标的','分类','模型框架','创建人','预测日期','测试值','预测频度','方向准确率','绝对偏差']
            ws_list.write_row(0, 0, headers)


            # ========= 详情页 =========
            ws_detail = workbook.add_worksheet('详情页')
            writer.sheets['详情页'] = ws_detail
            ws_detail.write(0, 0, target_name)
            ws_detail.write_row(1, 0, ['指标日期','实际值','预测值','方向','偏差率'])

            # ========= 日度数据表 =========
            ws_daily = workbook.add_worksheet('日度数据表')
            writer.sheets['日度数据表'] = ws_daily
            # 判断真实值列名是'真实值'还是'实际值'
            true_value_col = '真实值' if '真实值' in daily_df.columns else '实际值'
            daily_out = daily_df[['Date', true_value_col, TARGET_COL]].copy()
            daily_out.columns = ['指标日期','实际值','预测值']
            
            # 格式化日期为年/月/日
            daily_out['指标日期'] = pd.to_datetime(daily_out['指标日期']).dt.strftime('%Y/%m/%d')
            
            # 日度数据表不限制有效数字
            daily_out.to_excel(writer,sheet_name='日度数据表',index=False,header=False,startrow=2)

            ws_daily = writer.sheets['日度数据表']
            ws_daily.write(0, 0, target_name)
            ws_daily.write_row(1, 0, ['指标日期','实际值','预测值'])

    except Exception as e:
        print(f"生成Excel文件时发生错误: {str(e)}")
        raise
    else:
        print(f"已成功生成并填充 {output_path}")
