import numpy as np
import pandas as pd

# =========================
# 数据清洗与指标计算
# =========================

def preprocess_experiment_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    逻辑：
    1. 读取现有的 Fe_Remaining，将负数修正为 0.05。
    2. 读取现有的 Nd_Recovery。
    3. 重新计算 SF 和 log_SF。
    """

    df = df.copy()

    # =========================
    # Step 1: 修正 Fe Remaining
    # =========================
    if "Fe_Remaining" in df.columns:
        df["Fe_Remaining"] = pd.to_numeric(df["Fe_Remaining"], errors="coerce")
        df["Fe_Remaining"] = pd.to_numeric(df["Fe_Remaining"], errors="coerce")
        df.loc[df["Fe_Remaining"] < 0.5, "Fe_Remaining"] = 0.5
    else:
        print("⚠️ 警告：未找到 'Fe_Remaining' 列")

    if "Nd_Recovery" in df.columns:
        df["Nd_Recovery"] = pd.to_numeric(df["Nd_Recovery"], errors="coerce")
        df["Nd_Recovery"] = pd.to_numeric(df["Nd_Recovery"], errors="coerce")
        df.loc[df["Nd_Recovery"] > 100, "Nd_Recovery"] = 100
    else:
        print("⚠️ 警告：未找到 'Nd_Recovery' 列")

    # =========================
    # Step 2: 更新 Separation Factor (SF)
    # =========================
    if "Nd_Recovery" in df.columns and "Fe_Remaining" in df.columns:
        df["SF"] = df["Nd_Recovery"] / df["Fe_Remaining"]
        
        # =========================
        # Step 3: log(SF)
        # =========================
        df["log_SF"] = np.log(df["SF"])
    else:
        print("⚠️ 警告：缺少必要列，无法计算 SF")

    return df


# =========================
# Excel → Excel 主流程
# =========================

def clean_excel(
    input_xlsx: str,
    output_xlsx: str,
):
    """
    读取原始 Excel，清洗数据，并仅保留建模所需的特定列。
    """

    # 读取原始 Excel
    df_raw = pd.read_excel(input_xlsx)

    # 数据清洗
    df_clean = preprocess_experiment_data(df_raw)

    # =========================
    # 关键修改：只保留和你代码一致的列
    # =========================
    columns_to_keep = [
        "Con_pH",
        "Con_time (h)",
        "Fe_Remaining",
        "Nd_Recovery",
        "SF",
        "log_SF",
    ]

    # 筛选列（仅保留存在的列，防止报错）
    existing_cols = [col for col in columns_to_keep if col in df_clean.columns]
    df_export = df_clean[existing_cols]

    # 导出为新的 Excel
    df_export.to_excel(output_xlsx, index=False)

    print(f"✅ 清洗完成，已保存至: {output_xlsx}")
    print(f"📊 共导出 {len(df_export)} 条实验数据")
    print(f"📝 保留列: {existing_cols}")


# =========================
# 脚本入口
# =========================

if __name__ == "__main__":
    clean_excel(
        input_xlsx="All_data.xlsx",
        output_xlsx="cleaned_data_all.xlsx",
    )
