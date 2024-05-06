# %% Plot class imbalance
from unet_framework.utils.general_utils import calculate_class_occurrences_dataset, get_IDs_of_dir
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.dpi'] = 300

import unet_framework.data.Microtubules.BG_CM_MT.Training_data.Size_24x200x200.Target as target_dir_training

# Target directory training data
target_dir_training = target_dir_training.__path__._path[0]  # location of the target files (Training)


# File extension
target_ext = '.tif'  # target extension

# Target IDs
train_target_IDs = get_IDs_of_dir(target_dir_training, ext=target_ext)  # list of target files (Training)

occurences = calculate_class_occurrences_dataset(train_target_IDs, reduction='none') * 100

bg = occurences[:, 0]
cm = occurences[:, 1]
mt = occurences[:, 2]

fig1, ax1 = plt.subplots(figsize=(4, 4))
ax1.boxplot(occurences)
ax1.set_xticklabels(['Background', 'Cell membrane', 'Microtubules'])
ax1.set_ylim(0, 100)
ax1.set_ylabel('Presence in %')

ax1.set_title('Class occurrences')
import os

plt.style.use('seaborn-dark')
# plt.savefig(os.path.join('/data/s339697/master_thesis/Images', 'class_occurrences.png'), bbox_inches='tight')
plt.show()
