# **Escape-ELite-Network (EEN)**

## **Unveiling Viral Escape Mechanisms with Machine Learning**  
_A transformative approach to mutation analysis for SARS-CoV-2 and beyond._

Persistent viruses such as **Influenza**, **HIV**, and **Coronavirus** present significant challenges due to their ability to escape immune responses, hindering the development of lasting vaccines and treatments. This study introduces the **Escape Elite Network (EEN)**, an LSTM-based deep learning model that analyzes over **3.1 million viral spike protein sequences**, with a focus on **SARS-CoV-2**. EEN outperforms existing models in detecting escape mutations across various datasets, achieving high **AUC scores** and demonstrating statistical significance. Its ability to predict high-risk mutations before experimental validation positions it as a powerful tool for advancing vaccine and therapeutic development.


## **Execution Guide**  

### **Running Experimental Results for All Datasets**  
Execute the following script to generate experimental results:  
```bash
python Escape_score_predictor.py
```
This will run the function:  
```python
execute_aggregate_network()
```

### **Plotting AUC Curves**  
To visualize AUC plots, run:  
```python
plot_integrated_auc()
```

### **Reporting Additional Metrics**  
Compute and analyze additional metrics by executing:  
```python
compute_additional_metrics()
```
For a detailed report, verify results using the provided Jupyter notebook:  
```bash
jupyter notebook additional_metrics_reported.ipynb
