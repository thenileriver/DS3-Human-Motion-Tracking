import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

annotations = pd.read_csv('data.csv')


def plot_image(data, title = "cms"):
    """Plots image num in the list
    
    Args:
        data: The array where the image data is located
        img_num: The sample number in the dataset

    Returns:
        Plot of the image
    """
    fig = plt.figure
    plt.title(title)
    plt.imshow(data, cmap='gray')
    
    
    return plt.show() 

def make_sampling_vector(width, height, stride):
    """Create sampling vectors that define a grid.

    Args:
        width: Width of grid as an integer.
        height: Height of grid as an integer.
        stride: Steps between each sample.
    
    Returns:
        A pair of vectors xv, yv that define the grid.
    """
    xv = np.arange(0, width, stride, dtype="float32")
    yv = np.arange(0, height, stride, dtype="float32")
    return xv, yv


def make_confidence_map(x, y, xv, yv, sigma=1):
    """Make confidence maps for a point.

    Args:
        x: X-coordinate of the center of the confidence map.
        y: Y-coordinate of the center of the confidence map.
        xv: X-sampling vector.
        yv: Y-sampling vector.
        sigma: Spread of the confidence map.
    
    Returns:
        A confidence map centered at the x, y coordinates specified as
        a 2D array of shape (len(yv), len(xv)).
    """

    cm = np.exp(-(
    (xv.reshape(1, -1) - x) ** 2 + (yv.reshape(-1, 1) - y) ** 2
    ) / (2* sigma ** 2))
    return cm

  

def make_multi_nodal_cm(cord_array, img_height, img_width, stride, sigma):
    """Creates a confidence maps for n-nodes

    Args:
      cord_array: array containing elements of xy pairs
      img_height: height of input image
      img_width: width of input image

    Returns:
      Confidence maps aggregated nodes
    """

    output_cms = []

    for cord in cord_array:
      xv, yv = make_sampling_vector(img_width, img_height, stride)
      cord_cm = make_confidence_map(cord[0], cord[1], xv, yv, sigma)
      output_cms.append(cord_cm)

    return output_cms

def __getitem__(self, idx):
        """Return a single batch."""    
        i0 = idx * self.batch_size
        i1 = i0 + self.batch_size

        Y, X = [], []
        # for i in range(i0, i1):
        #     # lf = # TODO this should be images array formatted to (batch, img_height, img_width)  
            
        #     # for instance in lf:
        #     #     Y.append(make_multi_nodal_cm(instance, img_height=self.img_height, img_width=self.img_width, stride=self.output_stride, sigma=self.sigma))
        #     #     X.append(instance)
                                    
        # Stack the data into batches.
        X = np.stack(X, axis=0)
        Y = np.stack(Y, axis=0)
        
        return X, np.transpose(Y, (0, 2, 3, 1))

all_cms = []

# len(annotations.index())
for image_index in range(len(annotations.index)):
    cms = []
    # image_index = 1
    img_height = 200
    img_width = 200
    cords = []
    this_row = annotations.iloc[image_index]
    for col in range(2,33,2):
        cords.append([this_row.iloc[col],this_row.iloc[col+1]])
    # cords = [[30, 30],[100, 100], [130, 130]]
    sigma = 10
    stride = 2
    cms += make_multi_nodal_cm(cords, img_height, img_width, stride, sigma)
    all_cms.append(cms)

print(len(all_cms))
plot_image(all_cms[0][0])