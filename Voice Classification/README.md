# Voice classification

A deep learning model to classify voice using Pytorch
If you want to test any voice, first run make_dataset.ipynb to preprocess the voice and then you can inference 

---
## Installation
1 - Clone the repository
```
git clone https://github.com/Parisa-Bagherzadeh/pytorch.git
```
2 - Install requirements
```
pip install -r requirements.txt
```

## Inference

To inference , open up a temrinal and execute the following command :

```
python3 inference.py --voice filename of voice 
```
---
 <table>
     <tr>
       <td></td>
       <td>Accuracy</td>
       <td>Loss</td>
     </tr>
     <tr>
       <td>Train</td>
       <td>0.97</td>
       <td>1.57</td>
     </tr>
     <tr>
       <td>Test</td>
       <td> 0.88</td>
       <td>1.66</td>
     </tr>
   </table>
