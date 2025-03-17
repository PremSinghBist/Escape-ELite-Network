# **Escape-ELite-Network (EEN)**  

**Unveiling Viral Escape Mechanisms with Machine Learning**  
A transformative approach to mutation analysis for SARS-CoV-2 and beyond.  

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
