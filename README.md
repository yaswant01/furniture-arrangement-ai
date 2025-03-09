# 🏡 AI-Based Furniture Arrangement 🏚️
### An AI-powered tool to optimize furniture placement in a room based on user-defined constraints.

![ I have completed a project on AI Furniture Arrangement as a recruitment assessment for a company.](![Screenshot 2025-03-10 000615]![image](https://github.com/user-attachments/assets/d55ce4bf-61fc-4f74-b3e1-2ee994f32635)
)  <!-- Replace with an actual screenshot -->

---

## 📌 Project Overview
This project is an **AI-powered furniture arrangement system** that optimizes room layouts based on:
✔ **Room dimensions** (custom width & height)  
✔ **Furniture constraints** (types & sizes)  
✔ **Obstacles (e.g., walls, doors, windows)**  
✔ **Machine learning model trained to generate optimal furniture placements**  

🖥️ **Built with:**  
- **Python (TensorFlow/Keras)** for AI Model  
- **Streamlit** for Interactive UI  
- **Matplotlib & Seaborn** for Visualization  

---

## 🚀 Features
✅ **User-defined room dimensions** (e.g., 10x10, 15x20, etc.)  
✅ **Dynamic furniture placement based on AI predictions**  
✅ **Obstacle-aware layout suggestions**  
✅ **Intuitive & interactive Streamlit UI**  
✅ **Customizable furniture types and sizes**  

---

## 💪 Installation
### 1️⃣ Clone the Repository
```bash
git clone [https://github.com/yaswant01/furniture-arrangement-ai.git
cd furniture-arrangement-ai]
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🛠️ Usage
### 1️⃣ Generate a Synthetic Dataset
```bash
python dataset_generator.py
```
This creates `room_data.csv` and `room_labels.csv` for training.

### 2️⃣ Train the AI Model
```bash
python train_model.py
```
This will train and save the model as **`furniture_model.keras`**.

### 3️⃣ Run the Interactive Web App
```bash
streamlit run furniture_arrangement.py
```
This will launch a **web-based UI** where you can:
✔ Set **room dimensions**  
✔ Choose **furniture constraints**  
✔ Add **obstacles**  
✔ See AI-generated **optimized furniture placements**  

---

## 📸 Screenshots
### 1️⃣ Streamlit UI
![Streamlit UI](https://user-images.githubusercontent.com/example/ui_screenshot.png) <!-- Replace with an actual screenshot -->
![image](https://github.com/user-attachments/assets/6248da67-a504-4e80-807a-277cb62e846e)


### 2️⃣ AI-Suggested Furniture Placement
![AI Prediction](https://user-images.githubusercontent.com/example/prediction.png)  <!-- Replace with an actual screenshot -->
![image](https://github.com/user-attachments/assets/e95f671f-020c-4e9b-bc1c-5ad9a663bea3)


---

## 🔬 How It Works
1️⃣ **User Inputs**: Define **room size**, **furniture constraints**, and **obstacles** in the UI.  
2️⃣ **Data Processing**: The system **prepares input data** for AI prediction.  
3️⃣ **AI Model (Neural Network)**:
   - Uses a **trained model** to predict **optimal furniture placements**.
   - Takes into account **available space & obstacles**.
4️⃣ **Visualization**: The **optimized layout** is displayed using a **heatmap**.

---

## 📌 Model Details
- **Neural Network Architecture**
  - 3 Hidden Layers (256 → 128 → 64)
  - **ReLU Activation**
  - **Softmax Output**
  - **Binary Cross-Entropy Loss with Weighted Priority for Placements**
- **Training Data**
  - **Synthetic 10x10 room layouts**
  - **Randomly placed furniture & obstacles**
  - **Trained on 2000+ samples**

---

## 🌟 Example Use Cases
✔ **Interior Designers** - Quickly test different layouts before actual placement.  
✔ **Homeowners & Renters** - Optimize furniture placement for better space utilization.  
✔ **Game Developers** - Use AI-generated layouts for virtual simulations.  

---

## 🛠️ Future Improvements
🔹 **Support for irregular-shaped rooms**  
🔹 **Allow dragging/dropping furniture interactively**  
🔹 **Use Reinforcement Learning (Deep Q-Learning) for better spatial optimization**  

---

## 🙌 Contributing
We welcome contributions!  
🔹 Fork the repository  
🔹 Create a feature branch  
🔹 Submit a Pull Request  

---

## 📩 Contact
👤 **Yaswanth Dhulipalla**  
📧 yaswanthdhulipalla149@gmail.com  
📚 [LinkedIn](https://www.linkedin.com/in/yaswanth-dhulipalla-1920a724b/)  

---

## ⭐ If you found this useful, give it a star! ⭐
```
🌟 GitHub Repository: https://github.com/yaswant01/furniture-arrangement-ai ⭐
```

---
