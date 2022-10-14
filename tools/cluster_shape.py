from sklearn.cluster import KMeans
import os
import pickle
import numpy as np
import cv2


if __name__ == "__main__":
    # dataDir='/media/Seagate16T/tqminh/AmodalSeg/chicken_data/chicken_posture'
    dataDir='/data/tqminh/AmodalSeg/std_data/KINS_2'

    masks = None
    SHAPE_SIZE = 28
    with open(os.path.join(dataDir, 'shape_priors_{}.pkl'.format(SHAPE_SIZE)), 'rb') as f:
        masks = pickle.load(f)

    K = 20
    mask_features = masks.reshape((-1, SHAPE_SIZE*SHAPE_SIZE))
    kmeans = KMeans(n_clusters=K, random_state=0).fit(mask_features)
    with open(os.path.join(dataDir, 'k_shapes_{}.pkl'.format(SHAPE_SIZE)), 'wb') as f:
        pickle.dump(kmeans.cluster_centers_, f)

    try:
        os.mkdir(os.path.join(dataDir, 'mask_centers_{}/'.format(SHAPE_SIZE)))
    except:
        pass
    for i, center in enumerate(kmeans.cluster_centers_):
        center = np.reshape(center, (SHAPE_SIZE, SHAPE_SIZE))
        cv2.imwrite(os.path.join(dataDir, 'mask_centers_{}/center_{}.png'.format(SHAPE_SIZE, i)), center)

