import joblib
import pandas as pd
import os


def fast_predict(input_data_dict, model_path="automl_model.pkl"):
    """
    加载模型并对输入数据进行预测
    input_data_dict: 字典格式，例如 {"island": "Biscoe", "bill_length_mm": 45.1, ...}
    """
    try:
        # 1. 检查模型文件是否存在
        if not os.path.exists(model_path):
            return f"❌ 错误：在当前目录下找不到模型文件 {model_path}"

        # 2. 加载持久化的模型包
        data = joblib.load(model_path)

        model = data["model"]
        label_encoders = data["label_encoders"]
        task = data["task"]
        original_cols = data["original_cols"]

        # 3. 将输入转换为 DataFrame
        input_df = pd.DataFrame([input_data_dict])

        # 4. 应用训练时的 LabelEncoder 翻译逻辑
        for col in original_cols:
            if col in label_encoders:
                le = label_encoders[col]
                val = str(input_df[col].iloc[0])
                # 容错处理：如果输入了训练时没见过的值，默认转为第一类
                if val in le.classes_:
                    input_df[col] = le.transform([val])[0]
                else:
                    input_df[col] = 0

        # 5. 执行推理
        prediction = model.predict(input_df)[0]

        if task == "classification":
            prob = model.predict_proba(input_df)[0].max()
            return f"✅ 预测类别: {prediction} (置信度: {prob:.2%})"
        else:
            return f"✅ 回归预测值: {prediction:.2f}"

    except Exception as e:
        return f"❌ 推理失败: {str(e)}"


# --- 交付测试示例 ---
if __name__ == "__main__":
    # 打印模型基本信息，确保加载正常
    if os.path.exists("automl_model.pkl"):
        data = joblib.load("automl_model.pkl")
        print("--- 模型加载成功 ---")
        print(f"待输入特征: {data['original_cols']}")

        # 模拟一条测试数据（请根据你的数据集修改键名）
        test_sample = {col: 0 for col in data['original_cols']}
        # 例如对于企鹅数据集：test_sample = {"island": "Biscoe", "bill_length_mm": 40.0, ...}

        # print(fast_predict(test_sample))
    else:
        print("请确保目录下有 automl_model.pkl 文件")