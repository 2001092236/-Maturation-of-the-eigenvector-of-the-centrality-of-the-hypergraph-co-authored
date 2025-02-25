# Data-free distillation

Problem: distillate a model without the access to its training data (because it may be sensitive, like biometrics).
Idea: obtain some elements of data from the teacher and then perform distillation. So, we will try to recoder a train dataset and then will distillate the teacher into a student on such recovered train dataset. 

## Project Architecture

1. Hooked_Linear_Layer, Hoocked_Conv_Layer
   - have 'hooks' to get all activations
2. Hooked_FC
3. Hooked_LeNet
4. Hooked_AlexNet


## Public functions and methods
1. sample_item_from_last_logits_statistics
   - used to sample from a given class $C$ using gathered statistics about last logits
   - implements a gradient descent idea
2. sample_item_from_all_logits_statistics
   - used to sample from a given class $C$ using gathered statistics about all logits
   - implements a gradient descent idea
3. sample_adversarial
   - samples using adversarial mechanism given a student, a teacher and a generator
4. sample_inversion
   - samples an item using deep inversion mechanism
5. distillate
   - an ordinary distillation algorithm: given annotated dataset, teacher and student models distillates knowledge from teacher into a student
   

## Libraries and integration
1. PyPI 
2. PyTorch
3. Matplotlib
4. Transformers
5. Neptune (for logging)
6. Aquvitae (for distillation)
