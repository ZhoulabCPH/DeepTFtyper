# DeepTFtyper

Small cell lung cancer (SCLC) is a highly aggressive high-grade neuroendocrine carcinoma with a poor prognosis. Molecular subtyping has shown great potential in guiding treatment decisions. However, its clinical application remains limited due to the lack of sufficient samples and the complexity of molecular testing. In this study, we developed a graph neural network-based deep learning model (DeepTFtyper) for automatic image-based molecular subtype classification of SCLCs from H&E-stained whole-slide images (WSIs).

------

## Requirements

```
pip install -r requirements.txt
```

------

## Build Model 

You can start using the DeepTFtyper model to achieve molecular subtype prediction of SCLC by performing the following operations:

- Feature Extraction: run ./extractor/extract_feature.py
- Build Graph: run ./graph/build_graph.py
- Model Train and Test: run main.py

------

## Acknowledgments

- Building Graph Neural Networks Using the torch_geometric.