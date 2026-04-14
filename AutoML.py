import pandas as pd
import gradio as gr
from flaml import AutoML
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import io
import joblib
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 解决显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class UniversalAutoML:
    def __init__(self):
        self.automl = AutoML()
        self.task = None
        self.feature_names = None
        self.X_test = None
        self.y_test = None
        self.label_encoders = {}
        self.original_cols = None

    def get_columns(self, file_obj, url_path):
        try:
            if file_obj is not None:
                df = pd.read_csv(file_obj.name, nrows=5)
            elif url_path and url_path.strip():
                # 增加了简单的网络请求超时处理提示
                df = pd.read_csv(url_path.strip(), nrows=5)
            else:
                return []
            return list(df.columns)
        except Exception as e:
            print(f"列名获取失败: {e}")
            return []

    def train_system(self, file_obj, url_path, target_col, task_type, exclude_cols):
        try:
            if file_obj is not None:
                df = pd.read_csv(file_obj.name)
            elif url_path and url_path.strip():
                df = pd.read_csv(url_path.strip())
            else:
                return "❌ 错误：请提供数据源", None, None
        except Exception as e:
            return f"❌ 数据加载失败: {str(e)}", None, None

        if target_col not in df.columns:
            return f"❌ 错误：目标列 '{target_col}' 不存在，请检查大小写。", None, None

        if exclude_cols:
            actual_excludes = [c for c in exclude_cols if c != target_col]
            df = df.drop(columns=actual_excludes)

        self.task = task_type.lower()
        self.label_encoders = {}
        df = df.fillna(df.mode().iloc[0])
        X = df.drop(columns=[target_col])
        y = df[target_col]
        self.original_cols = list(X.columns)

        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le

        self.feature_names = list(X.columns)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_test, self.y_test = X_test, y_test
        self.automl.fit(X_train=X_train, y_train=y_train, task=self.task, time_budget=15, verbose=0)

        fi_plot = self._plot_feature_importance()
        cm_plot = self._plot_confusion_matrix() if self.task == "classification" else None
        return "✅ 训练成功", cm_plot, fi_plot

    def _plot_feature_importance(self):
        importances = self.automl.feature_importances_
        feature_imp = pd.DataFrame({'Value': importances, 'Feature': self.feature_names}).sort_values(by="Value",
                                                                                                      ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Value", y="Feature", data=feature_imp, palette="viridis", hue="Feature", legend=False)
        plt.title("模型特征贡献度分析")
        plt.tight_layout()
        return self._plt_to_img()

    def _plot_confusion_matrix(self):
        y_pred = self.automl.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('预测类别');
        plt.ylabel('真实类别')
        plt.title('分类混淆矩阵');
        plt.tight_layout()
        return self._plt_to_img()

    def _plt_to_img(self):
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        return img

    def predict(self, *args):
        if not self.automl.model: return "请先训练", None
        input_data = list(args)
        for i, col_name in enumerate(self.original_cols):
            if col_name in self.label_encoders:
                le = self.label_encoders[col_name]
                try:
                    input_data[i] = le.transform([str(input_data[i])])[0]
                except:
                    input_data[i] = 0

        input_df = pd.DataFrame([input_data], columns=self.feature_names)
        if self.task == "classification":
            pred = self.automl.predict(input_df)[0]
            probs = self.automl.predict_proba(input_df)[0]
            plt.figure(figsize=(8, 4))
            sns.barplot(x=probs, y=self.automl.classes_, palette="magma")
            plt.title(f"预测判定: {pred}");
            plt.tight_layout()
            return f"🚀 判定结果: {pred}", self._plt_to_img()
        else:
            pred = self.automl.predict(input_df)[0]
            return f"📈 预测值: {pred:.2f}", None

    def save_assets(self):
        model_path, report_path = "automl_model.pkl", "模型技术名片.md"
        save_data = {"model": self.automl, "task": self.task, "feature_names": self.feature_names,
                     "label_encoders": self.label_encoders, "original_cols": self.original_cols}
        joblib.dump(save_data, model_path)

        best_algo = self.automl.best_estimator if hasattr(self.automl, "best_estimator") else "AutoML"
        md_content = f"# 🤖 交付级模型技术报告\n\n- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n- **核心算法**: `{best_algo}`\n- **输入维度**: {len(self.feature_names)} 项\n\n### 特征列表\n" + "\n".join(
            [f"- {c}" for c in self.feature_names])
        with open(report_path, "w", encoding="utf-8") as f: f.write(md_content)
        return [model_path, report_path]


engine = UniversalAutoML()
with gr.Blocks(title="AutoML Pro") as demo:
    gr.Markdown("# AutoML 自动化建模平台")
    feature_list = gr.State([])

    with gr.Tab("1. 自动化流水线"):
        with gr.Row():
            file_input = gr.File(label="上传本地 CSV")
            url_input = gr.Textbox(label="数据 URL (输入后按回车刷新)",
                                   placeholder="https://raw.githubusercontent.com/...")

        column_selector = gr.CheckboxGroup(label="剔除干扰特征", choices=[])


        # 【修复重点】：定义统一的刷新函数
        def refresh_columns(f, u):
            cols = engine.get_columns(f, u)
            return gr.update(choices=cols, value=[])


        # 绑定多个触发源：本地文件改变、URL回车、URL失去焦点
        file_input.change(refresh_columns, [file_input, url_input], column_selector)
        url_input.submit(refresh_columns, [file_input, url_input], column_selector)
        url_input.blur(refresh_columns, [file_input, url_input], column_selector)

        with gr.Row():
            target_input = gr.Textbox(label="目标列名", value="species")
            task_input = gr.Radio(["Classification", "Regression"], label="任务类型", value="Classification")

        train_btn = gr.Button("🔥 开启全自动训练", variant="primary")
        status_out = gr.Textbox(label="运行日志")
        download_files = gr.File(label="📥 下载交付包 (.pkl & .md)", file_count="multiple")

        with gr.Row():
            plot_cm = gr.Image(label="混淆矩阵")
            plot_fi = gr.Image(label="特征权重")

    with gr.Tab("2. 智能推理终端"):
        @gr.render(inputs=feature_list)
        def render_predict(ui_config):
            if not ui_config: return gr.Markdown("### ⏳ 请先完成模型训练")
            inputs = []
            with gr.Column():
                for i, (name, is_cat, choices) in enumerate(ui_config):
                    if is_cat:
                        inputs.append(gr.Dropdown(label=name, choices=choices, value=choices[0]))
                    else:
                        inputs.append(gr.Number(label=name, value=0))
                p_btn = gr.Button("⚡ 立即预测", variant="primary")
                res_t = gr.Textbox(label="预测结论")
                res_i = gr.Image(label="决策依据 (概率分布)")
                p_btn.click(engine.predict, inputs=inputs, outputs=[res_t, res_i])


    def train_wrapper(file, url, target, task, excludes):
        msg, cm, fi = engine.train_system(file, url, target, task, excludes)
        paths = engine.save_assets()
        ui_cfg = []
        if engine.original_cols:
            for col in engine.original_cols:
                is_cat = col in engine.label_encoders
                choices = list(engine.label_encoders[col].classes_) if is_cat else []
                ui_cfg.append((col, is_cat, choices))
        return msg, cm, fi, ui_cfg, paths


    train_btn.click(train_wrapper, [file_input, url_input, target_input, task_input, column_selector],
                    [status_out, plot_cm, plot_fi, feature_list, download_files])

if __name__ == "__main__":
    demo.launch()