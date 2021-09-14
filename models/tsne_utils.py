from posixpath import sep
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

import os


import pickle as pkl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import PCA
import sklearn
import time
import cv2
from tqdm import tqdm

from glob import glob
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.preprocessing import StandardScaler
from matplotlib import animation
from termcolor import colored

#import scipy.spatial.distance.pdist as pom

label_to_id_dict = {v: i for i, v in enumerate(['not_infected', 'infected'])}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}


def _3d_vizualize_features( tsne_result_scaled,  label_ids,  data_dir=''):
    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(111,projection='3d')

    plt.grid()
        
    nb_classes = len(np.unique(label_ids))

    #print(colored(np.unique(label_ids), 'red'))


        
    for label_id in np.unique(label_ids):
        ax.scatter(tsne_result_scaled[np.where(label_ids == label_id), 0],
                    tsne_result_scaled[np.where(label_ids == label_id), 1],
                    tsne_result_scaled[np.where(label_ids == label_id), 2],
                    alpha=0.8,
                    color= plt.cm.viridis(float(label_id)),
                    marker='o',
                    label=label_id)
    ax.legend(loc='best')
    ax.view_init(25, 45)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_zlim(-2.5, 2.5)

    anim = animation.FuncAnimation(fig, lambda frame_number: ax.view_init(30, 4 * frame_number), interval=75, frames=90)

    plot_3d_animation_filename = os.path.join(data_dir, 'animation.avi')
    #anim.save(plot_3d_animation_filename, writer='imagemagick')


    anim.save(plot_3d_animation_filename, fps=10)

    return


def visualize_scatter_with_images(X_2d_data, images,  image_zoom=1.0,  data_dir=''):
    fig, ax = plt.subplots(figsize=(45,45))
    artists = []
    for xy, i in zip(X_2d_data, images):
        x0, y0 = xy
        img = OffsetImage(i, zoom=image_zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(X_2d_data)
    ax.autoscale()
    plt.savefig(os.path.join(data_dir, 'tsne_features_images.png'), dpi=600)


def tsne_scattering(tsne_results, df_subset, data_dir, nb_classes):
    print(tsne_results.shape)
    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("tab10", nb_classes),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    plt.savefig(os.path.join(data_dir, 'tsne_features.png'), dpi=600)

def plot_with_pca(dir_name, all_data, list_images, all_target):
    all_target = np.asarray(all_target)
    list_img_array = []


    for cp, _ in tqdm(enumerate(list_images)):
        image_path = list_images[cp]
        list_img_array.append(cv2.resize(cv2.imread(image_path, 1), dsize=(32, 32)))



    label_ids = all_target

    nb_classes = len(np.unique(label_ids))
    
    print(all_data.shape)
    print(all_target.shape)

    all_data = np.squeeze(all_data)
    
    
        
    nb_features = all_data.shape[1] 
    feat_cols = [ 'feature_'+str(i) for i in range(nb_features) ]
    
    df = pd.DataFrame(all_data, columns=feat_cols)
    
    df['y'] = all_target
    
    print( np.count_nonzero( all_target))
    
    df['label'] = df['y'].apply(lambda i: str(i))
    
    print(sorted(sklearn.neighbors.VALID_METRICS['brute']))
    
    
    distance_metric = 'l2'
    # perform t-SNE embedding
    #vis_data = bh_sne(all_data)
    
    
    
    #nb_videos = 1000
    
    nb_videos = all_data.shape[0]
    
    # For reproducability of the results
    np.random.seed(42)
    rndperm = np.random.permutation(nb_videos)
    
    df_subset = df.loc[rndperm[:nb_videos],:].copy()
    data_subset = df_subset[feat_cols].values
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data_subset)
    


    df_subset['pca-one'] = pca_result[:,0]
    df_subset['pca-two'] = pca_result[:,1] 
    df_subset['pca-three'] = pca_result[:,2]
    
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    
    
    
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40,
                n_iter=400, n_jobs = 8, metric=distance_metric)
    
    tsne_results = tsne.fit_transform(data_subset)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    
    print(tsne_results.shape)
    
    
    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]

    tsne_results = tsne.fit_transform(data_subset)
    tsne_result_scaled = StandardScaler().fit_transform(tsne_results)

    tsne_scattering(tsne_results, df_subset, dir_name, nb_classes)
    visualize_scatter_with_images(tsne_result_scaled, list_img_array, image_zoom=0.7, data_dir=dir_name)


    tsne = TSNE(n_components=3)
    tsne_result = tsne.fit_transform(pca_result)
    tsne_result_scaled = StandardScaler().fit_transform(tsne_result)


    _3d_vizualize_features( tsne_result_scaled,  label_ids,  data_dir=dir_name)
    
