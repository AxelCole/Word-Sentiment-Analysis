1 Introduction

This README accompanies a research project in Natural Language Processing
(NLP), specifically focusing on sentiment analysis. The objective of this project
is to determine the sentiment (positive, neutral, negative) of a specific word
within a given sentence. It’s worth noting that the dataset available for this
task is limited in size. Consequently, to address this challenge, deep learning
approaches were considered as a viable solution.
The model selected for this task is based on the DistilBERT architecture,
available through the Hugging Face library. DistilBERT is a version of the
BERT (Bidirectional Encoder Representations from Transformers) model. The
choice of DistilBERT was motivated by its efficiency in handling large-scale
language understanding tasks while maintaining memory efficiency, which is
crucial given the constraints of limited dataset size and computational resources.

2 Methodology

In this research, two main approaches were studied and compared: with LoRA
and without LoRA.
In adherence to the guidelines and constraints of this project, we have strictly
utilized only the dependencies and libraries permitted within the scope of our
work.
The metric we wanted to optimize was the accuracy of the model. Accuracy
is a crucial metric for optimizing model performance in this context, as it directly
reflects the model’s ability to correctly classify sentiments.

2.1 Approach with LoRA

LoRA in Deep Learning adjusts the relevance of information at different layers
to prevent overfitting and improve generalization. In sentiment analysis, LoRA
helps the model focus on relevant features, enhancing performance with limited
data.

The LoRA architecture offers several advantages and specific adjustments:

• Rank Adjustment to Avoid Overfitting: LoRA adjusts the rank of back-
propagation matrices to limit overfitting and promote generalization.
• Scaling Factor Adjustment: LoRA also allows adjustment of the scaling
factor to optimize performance.
• Fast Convergence: Experiments have shown rapid convergence in less than
5 epochs, typically by the 3rd epoch.
• Memory Efficiency: LoRA requires less memory to store gradients, making
it more resource-efficient.

However, despite these advantages, the approach with LoRA presented slightly
lower performance with an accuracy of 83% and therefore we do not give much
more details on it.

2.2 Approach without LoRA

In this approach, the entire network is fine-tuned for 1 epoch, followed by fine-
tuning only the classifier for 20 epochs (= the head). By freezing the pre-trained
layers, we preserve the knowledge learned from a large dataset, which helps
capture general language patterns. Fine-tuning only the classifier allows us to
adjust the model specifically for sentiment analysis without risking overfitting
or losing the learned representations in the earlier layers.

• Better Performance: Without LoRA, better performance was achieved
with an accuracy of 84% (up to a maximum of 85.5%).
• Stability: The approach without LoRA appeared to be more stable.
• Slower Execution: However, it was slower with an execution time of
234s/run.
• Increased Memory Requirement: More memory was necessary to store
gradients for epoch 1 (though we are still far under the 14 GB allowed on
GPU).
• Learning Rate Adjustment: Learning rate was increased during the second
phase with 20 epochs to accelerate learning as it was slower otherwise.
Learning rate was adjusted from 5e-5 to 2e-4.
• Two-phase Training: The first phase adapted the entire network to the
new classification task and data, while the second phase fine-tuned only
the head (classifier) without overfitting the training set. Otherwise, the
validation loss increased quickly if further fine-tuning of the entire network
continued in subsequent epochs.

3 Conclusion

Our main model does not utilize LoRA because our primary focus is achieving
the most effective results in sentiment analysis. By fine-tuning only the classifier
and freezing the other layers during most of the training phase, we optimize
performance while adapting the model specifically to the sentiment analysis
task.
Although LoRA offers advantages, it yielded a slightly lower accuracy com-
pared to the approach of freezing the last layers. This discrepancy can be
attributed to the nuanced trade-off between model complexity and task speci-
ficity. By fine-tuning only the classifier, we maintain the model’s ability to
capture general language patterns from pre-trained layers while tailoring its
predictive capabilities to sentiment analysis.

Note: Our best model (without LoRA) finds has an average accuracy of 84% (and sometimes up to 85.5%)

PS : Automatic Mixed Precision training is integrated to classifier_LoRA_AMP.py and allows to go 2x faster during training.

