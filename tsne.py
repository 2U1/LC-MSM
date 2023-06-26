from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


embeddings = np.load('seg_embeddings.npy')
labels = np.load('seg_labels.npy')

np.random.seed(5646)
uni_labels = np.unique(labels)

overall_embeddings = []
overall_labels = []

for key in uni_labels:
    index = (labels == key).squeeze(axis=-1)
    num = index.sum().item()
    class_embed = embeddings[index]
    class_labels = labels[index]

    index = np.random.choice(num, size=1000, replace=False)
    overall_embeddings.append(class_embed[index])
    overall_labels.append(class_labels[index])

overall_embeddings = np.concatenate(overall_embeddings, axis=0)
overall_labels = np.concatenate(overall_labels, axis=0)

np.save('seg_class_embed.npy', overall_embeddings)
np.save('seg_class_label.npy', overall_labels)

print(overall_embeddings.shape)
print(overall_labels.shape)


matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 30

# size: (sample_num, embedding_dim)
net1_embeddings = np.load('seg_class_embed.npy')
net1_target = np.load('seg_class_label.npy')  # size: (sample_num, )

net1_target = net1_target.squeeze(axis=-1)
target_value = list(set(net1_target))


color_dict = {}
# colors = ['black', 'red', 'gold', 'green', 'orange', 'pink', 'magenta', 'slategray', 'greenyellow', 'lightgreen',
#           'brown', 'chocolate', 'mediumvioletred', 'navy', 'lightseagreen', 'aqua', 'olive', 'maroon', 'yellow']


colors = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
          [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
          [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
          [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
          [0, 80, 100], [0, 0, 230], [119, 11, 32]]

new_color = []

for color in colors:
    color[0] = round(color[0]/255, 1)
    color[1] = round(color[1]/255, 1)
    color[2] = round(color[2]/255, 1)

    new_color.append((color[0], color[1], color[2]))

class_names = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

for i, t in enumerate(target_value):
    color_dict[t] = new_color[i]
print(color_dict)

net1 = TSNE(early_exaggeration=100).fit_transform(net1_embeddings)
np.save('tsne.npy', net1)

net1 = np.load('tsne.npy')

for i in range(len(target_value)):
    tmp_X1 = net1[net1_target == target_value[i]]
    plt.scatter(tmp_X1[:, 0], tmp_X1[:, 1], color=color_dict[target_value[i]],
                marker='o', s=1, label=class_names[i])

plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig("tsne.png")
