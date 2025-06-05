import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import functions as fn

st.title("Handwriting recognition")
option = st.sidebar.radio("",
    ["Home", "💻 Train"],
    label_visibility="collapsed"
)
@st.cache_data
def load_npz(file):
    data = np.load(file)
    if 'images' in data and 'labels' in data:
        return data['images'], data['labels']
    else:
        return None, None

# ---------- TRAIN PAGE ----------
if option == "💻 Train":
    with st.expander("Dataset"):
        dataset = st.file_uploader("Upload dataset (.npz)", type=["npz"])
        data_info = False

        if dataset is not None:
            X, y = load_npz(dataset)
            if X is not None:
                num_classes = len(np.unique(y))
                data_info = True
                st.success(f"Loaded {y.shape[0]} samples. Shape: {X.shape[1:]}. Classes: {num_classes}", icon="✅")
            else:
                st.error("Invalid file: Missing 'images' or 'labels' keys.")

        view_data = st.toggle("View dataset")
        if view_data and data_info:
            with st.spinner("Running..."):
                fig, axs = plt.subplots(num_classes, 10, figsize=(20, num_classes * 2))
                for i in range(num_classes):
                    idxs = np.where(y == i)[0]
                    for j in range(10):
                        target = np.random.choice(idxs)
                        axs[i][j].axis("off")
                        axs[i][j].imshow(X[target].reshape(32, 32), cmap="gray")
                st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        epochs = int(st.text_input("Epochs", "10"))
    with col2:
        test_size = float(st.text_input("Test Size", "0.2"))

    if st.button("Train", use_container_width=True):
        if not data_info:
            st.warning("Please upload a valid dataset first.", icon="⚠️")
        else:
            model, train_history, test_history, train_acc, test_acc = fn.fit_model(X, y, test_size, epochs, num_classes)
            st.session_state["model"] = model

            st.success(f"Train Accuracy: {train_acc:.2f}%. Test Accuracy: {test_acc:.2f}%", icon="✅")

            # Plot training history
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.plot(train_history.history['loss'], label="Loss", color="blue")
            plt.plot(test_history.history['loss'], label="Val Loss", color="cyan")
            plt.xlabel("Epochs")
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(train_history.history['accuracy'], label="Accuracy", color="red")
            plt.plot(test_history.history['accuracy'], label="Val Accuracy", color="pink")
            plt.xlabel("Epochs")
            plt.legend()
            st.pyplot(plt)

# ---------- HOME PAGE ----------
if option == "Home":
    st.write("🖌️ Draw a character below:")

    canvas_result = st_canvas(
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        height=150,
        width=150,
        drawing_mode="freedraw",
        key="canvas"
    )

    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
        img = img.convert("L").resize((32, 32))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, -1)

        model = st.session_state.get("model", None)
        if model is not None and st.button("Predict"):
            X_pred = model.predict(img_array)
            y_pred = np.argmax(X_pred, axis=1)[0]
            top_5 = np.argsort(X_pred[0])[::-1][:5]
            st.image(img_array.reshape(32, 32), caption="Processed Image (32x32)", use_container_width=False)
            for i in top_5:
                label = chr(65 + i)
                confidence = X_pred[0][i] * 100
                st.write(f"{label}: {confidence:.2f}%")