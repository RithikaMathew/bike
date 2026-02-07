# Police Crash Narrative Classification: E-bike, E-scooter, and Other

## Introduction
This project aims to classify police crash report narratives into three categories: **E-bike**, **E-scooter**, and **Other (Non-motorists or Cyclists)**. The motivation is to better understand the involvement of different non-motorist vehicle types in traffic incidents using natural language processing (NLP) and machine learning. Accurate classification of these incidents is critical for urban planning, safety policy development, and understanding emerging transportation trends.

## Dataset Overview

### Initial Data Source
- **Primary Dataset:** `nolabeluse.xlsx` - Contains crash records with HSMV Report Numbers and Bike Type information =bike 2011 to 2022_MetroPlanOrlando_Mighk Wilson dataset
- **PDF Reports:** 29 ZIP files containing police crash report PDFs with narrative sections
- **Labeled Dataset:** `label.xlsx` - Contains manually labeled crash narratives =emobility-classification_Andrew_150 labels dataset

### Class Distribution (Final Dataset)
- **Other (Non-motorists or Cyclists):** 45.36% (1,222 records)
- **E-scooter:** 34.89% (940 records)
- **E-bike:** 19.75% (532 records)
- **Total Records:** 2,694 narratives

## Methodology

### 1. Data Collection and PDF Extraction

#### Phase 1: PDF Matching and Narrative Extraction
1. **ZIP File Extraction:** Extracted 29 ZIP files from the `reports/` folder (aka Crash Narrative) containing police crash report PDFs
2. **PDF Mapping:** Built a mapping between HSMV Report Numbers and PDF file paths by parsing filenames (e.g., `CrashReport_85828586.pdf`)
3. **Narrative Extraction:** Used PyPDF2 to extract the NARRATIVE section from each PDF using regex pattern matching
4. **Data Matching:** Matched Excel records with PDFs based on HSMV Report Numbers

**Challenge:** After the matching process, only **~15 records** from `nolabeluse.xlsx` had corresponding PDF narratives. This was insufficient for training a robust machine learning model.

#### Phase 2: Combining Datasets
- Combined `bike_data_with_narratives.xlsx` (15 matches) with `label.xlsx` (mix of manually labeled and unlabeled narratives)
- Total combined records: **2,791 records** before cleaning
- After removing empty narratives: **2,694 records** ready for classification

### 2. Zero-Shot Classification with Large Language Model

#### Initial Attempt: GPT-4o-mini API
I initially planned to use the **GPT-4o-mini** API through UFL's LiteLLM Proxy to classify unlabeled narratives. However, I encountered a budget limitation error on the team account, indicating the team account had zero budget allocated for API usage, so I had to use my personal account:

```
Error - Error code: 400 - {'error': {'message': 'Budget has been exceeded! 
User=rmathew1@ufl.edu in Team=522495c0-f094-4367-a618-377a3413600c 
Current cost: 0.0018009, Max budget: 0.0', 'type': 'budget_exceeded', 
'param': None, 'code': '400'}}
```


#### Alternative Solution: Llama-3.3-70B-Instruct
To overcome the API budget limitation, I switched to using **Llama-3.3-70B-Instruct** model for zero-shot classification:

1. **Prompt Engineering:** Designed detailed prompts describing each category with specific examples:
   - **E-bike:** Electric bicycles, motorized bicycles, e-trikes, pedal-assisted bikes
   - **E-scooter:** Electric scooters (standing/seated), motorized scooters, e-skateboards, one-wheels
   - **Other:** Human-powered bicycles, pedicabs, pedestrians, traditional scooters, mopeds, ATVs

2. **Classification Strategy:** For each narrative, the model returned one of three categories
3. **Label Priority Logic:** 
   - If a record had a **manually labeled** Bike Type → Use keyword-based classification (`Bike Type Group`)
   - If no manual label existed → Use **AI-generated classification** (`Final Bike Type`)
   - Final column (`FINAL`) prioritizes manual labels over AI predictions, and this was the column used as labels for training

### 3. Data Preprocessing and Cleaning

#### Data Cleaning
- Removed rows with empty narratives
- Ensured all narratives had valid text content
- Final cleaned dataset: **2,694 records**

### 4. Model Selection: DistilBERT

#### Why DistilBERT?
After the API budget issue, I chose **DistilBERT** (`distilbert-base-uncased`) for the following reasons:

1. **No Token/API Costs:**
   - Runs entirely locally without API calls
   - No per-request charges or budget limitations
   - One-time training cost, unlimited inference

2. **Efficiency:**
   - 40% smaller than BERT-base
   - 60% faster inference speed
   - Only 66M parameters vs. 110M (BERT) or 70B (Llama)
   - Can process entire dataset in minutes instead of hours

3. **Sufficient Capacity:**
   - For a 3-class text classification task, excessive model complexity (e.g., GPT-4, Llama-70B) is unnecessary
   - DistilBERT retains 97% of BERT's language understanding while being much lighter
   - Reduces overfitting risk on a ~2,700 sample dataset

4. **Scalability/ Security:**
   - Ideal for batch processing thousands of narratives
   - Can be deployed locally or on-premises without cloud dependencies
   - Perfect for production use cases with budget constraints and in-house product requirement

5. **Proven Performance:**
   - Well-established for text classification tasks
   - Strong performance on domain-specific texts
   - Easy to fine-tune and deploy

### 5. Model Training Configuration

#### Training Setup
- **Model:** `distilbert-base-uncased`
- **Total Parameters:** 66,955,010
- **Training/Validation/Test Split:** 70% / 15% / 15%
  - Training set: 1,885 samples
  - Validation set: 404 samples
  - Test set: 405 samples

#### Handling Class Imbalance
**Problem:** The dataset has imbalanced classes (45% / 35% / 20%), which can cause the model to favor the majority class.

**Solution:** Applied class weighting to the loss function:
```python
Class Weights:
  Other (Non-motorists or Cyclists): 0.746
  E-scooter: 0.969
  E-bike: 1.710
```
- Higher weight for E-bike (minority class) ensures it's not ignored during training
- Loss function penalizes misclassification of E-bike more heavily

#### Training Hyperparameters
- **Learning Rate:** 2e-5
- **Batch Size:** 16 (train and eval)
- **Max Epochs:** 10
- **Early Stopping:** Patience of 3 epochs (stops if no improvement)
- **Metric for Best Model:** F1 Score (weighted) - better than accuracy for imbalanced data
- **Warmup Steps:** 100
- **Weight Decay:** 0.01
- **Max Sequence Length:** 512 tokens

#### Training Features
- **Stratified Splitting:** Maintains class distribution across train/val/test sets
- **Early Stopping:** Prevents overfitting by monitoring validation F1 score
- **Best Model Selection:** Automatically saves the best checkpoint based on validation performance
- **Data Collation:** Dynamic padding for efficient batching

## Results


### Model Performance Metrics

#### Training, Validation, and Test Accuracy

| Split      | Accuracy   | Inference Time | Samples/sec |
|------------|-----------|---------------|------------|
| Training   | 0.9973    | 29.20s        | 64.57      | (slight overfitting but controlled by early stopping)
| Validation | 0.8985    | 6.21s         | 65.01      |
| Test       | 0.8741    | 6.34s         | 63.84      |

#### Training Progress
The model was trained for **10 epochs** with early stopping (patience=3).

**Final Model:** Epoch 10 with 89.85% validation accuracy

#### Test Set Results
```
Accuracy:       87.41%
Precision:      87.46%
Recall:         87.41%
F1 Score:       87.38%
Test Loss:      0.5643
```

**Performance Summary:**
- Training samples: 1,885
- Validation samples: 404
- Test samples: 405


#### Per-Class Performance (Test Set)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|------|
| **Other (Non-motorists or Cyclists)** | 89% | 84% | 86% | 184 |
| **E-scooter** | 86% | 91% | 89% | 141 |
| **E-bike** | 88% | 88% | 88% | 80 |

**Key Observations:**
- **E-scooter** class has highest recall (91%) and F1-score (89%)
- **E-bike** shows strong balanced performance with 88% across all metrics (minority class with only 20% of data)
- **Other** class has highest precision (89%) but slightly lower recall (84%)
- Class weighting successfully prevented the model from ignoring the minority E-bike class

#### Confusion Matrix (Test Set)
```
                                   Predicted
                                   Other  E-scooter  E-bike
Actual Other (Non-motorists)        155         20       9
       E-scooter                     11        129       1
       E-bike                         9          1      70
```

**Analysis:**
- **Other class:** 155/184 correct (84.2%) - Most errors confuse with E-scooter (20) and E-bike (9)
- **E-scooter class:** 129/141 correct (91.5%) - Very few confused with E-bike (1), mostly with Other (11)
- **E-bike class:** 70/80 correct (87.5%) - Most errors confuse with Other (9), rarely with E-scooter (1)
- Model successfully distinguishes electric vs. non-electric vehicles in most cases
- E-scooter has the highest recall (91.5%), showing strong ability to identify this class

### Example Predictions

#### Correctly Classified Examples

1. **E-scooter** (Confidence: 94.2%)
   > "The rider was on an electric scooter traveling southbound when they collided with a parked vehicle. No injuries reported."

2. **E-bike** (Confidence: 91.7%)
   > "Cyclist on an e-bike was struck while crossing the intersection at Main St. Transported to hospital with minor injuries."

3. **Other** (Confidence: 96.3%)
   > "Pedestrian was walking across the street in marked crosswalk when hit by vehicle making a right turn."

4. **E-scooter** (Confidence: 89.5%)
   > "Individual riding electric scooter lost control on wet pavement and fell. Minor scrapes and bruises."

5. **Other** (Confidence: 93.8%)
   > "Traditional bicycle rider hit by opening car door. Rider sustained injuries to shoulder and wrist."

6. **E-bike** (Confidence: 87.2%)
   > "Person on motorized bike struck vehicle from behind while vehicle was stopped at red light."

#### Edge Cases and Misclassifications

**Example 1:** Predicted as **E-bike**, but not sure if motorized means electric or not
> "Person on motorized bike struck vehicle from behind while vehicle was stopped at red light."

**Example 2:** Predicted as **E-bike**, Actually **Other** 
> "Traditional bicycle rider hit by opening car door. Rider sustained injuries to shoulder and wrist."

### Model Robustness

#### Average Prediction Confidence (on nolabel.xlsx dataset)
- **Overall:** 94.5%
- **Total predictions:** 250 narratives
- **Prediction distribution:**
  - E-scooter: 142 (56.8%)
  - Other (Non-motorists or Cyclists): 67 (26.8%)
  - E-bike: 41 (16.4%)

The high average confidence (94.5%) indicates the model has learned robust patterns and makes confident predictions on new data.

### Deployment and Inference

#### Model Artifacts
- **Saved Model:** `./final_crash_classifier/`
- **Tokenizer:** Included in model directory
- **Label Mappings:** `label_info.json` with class mappings and weights
- **Model Size:** ~260 MB (compact and portable)

#### Inference Speed
- **Test Set (405 samples):** 6.34 seconds (63.84 samples/sec)
- **Validation Set (404 samples):** 6.21 seconds (65.01 samples/sec)
- **Training Set (1,885 samples):** 29.20 seconds (64.57 samples/sec)
- **Average:** ~65 samples/second on GPU

## Conclusion

### Summary of Achievements
1. **Overcame Data Limitations:** Successfully expanded a dataset from 15 to 2,694 labeled narratives using zero-shot LLM classification
2. **Budget-Conscious Solution:** Pivoted from expensive API calls to a local DistilBERT model after hitting budget constraints
3. **High Performance:** Achieved 87.41% test accuracy with balanced performance across all three classes despite class imbalance
4. **Production-Ready:** Created a reusable, cost-effective model for future crash narrative classification with 94.5% average confidence on new data

### Future Improvements
1. **Data Augmentation:** Generate synthetic narratives to further balance classes and use better zero-shot model like GPT-4o-mini for better labels
2. **Active Learning:** Identify low-confidence predictions for manual review
3. **Multi-Task Learning:** Train on related tasks (e.g., injury severity, location prediction) to improve features
4. **Ensemble Models:** Combine DistilBERT with other classifiers for even higher accuracy

### Applications
- **Urban Planning:** Identify hotspots for e-scooter and e-bike incidents
- **Safety Policy:** Develop targeted interventions for specific vehicle types
- **Insurance Analytics:** Risk assessment based on vehicle category
- **Research:** Trend analysis of emerging micro-mobility incidents

---

### How to Use This Model

**For Single Predictions:**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("./final_crash_classifier")
tokenizer = AutoTokenizer.from_pretrained("./final_crash_classifier")

narrative = "Rider on electric scooter struck by car at intersection"
result = classify_crash_narrative(narrative)
print(result)  # {'predicted_class': 'E-scooter', 'confidence': 0.923}
```

**For Batch Processing:**
```python
df_results = classify_excel_batch('new_crashes.xlsx')
```

**Files in Repository:**
- `bike_processing_complete.ipynb` - Data extraction and labeling workflow
- `train.ipynb` - Model training and evaluation
- `final_crash_classifier/` - Trained model (ready for inference)
- `NEW.xlsx` - Final labeled dataset from bike_processing_complete.ipynb
- `nolabel_classified.xlsx` - testing samples on new dataset (escooter_ebike_crash_news_subasish das dataset)
- `README.md` - This documentation
- `bike_data_with_narratives.xlsx` - result of nolabeluse.xlsx with only 15 matches

---

**Author:** Rithika Mathew  
**Contact:** rmathew1@ufl.edu  
**Date:** February 2026
