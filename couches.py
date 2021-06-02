import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from skimage import data

from imageio import imread
import glob


matplotlib.rcParams['font.size'] = 18


# In[ ]:


def imshow(img, title=''):
    plt.title(title)
    if img.ndim == 2:
        plt.imshow(img, cmap=plt.cm.gray)
    else:
        if img.shape[2] == 1:
          img = np.repeat(img, repeats=3, axis=2)
        plt.imshow(img)
    
    plt.show()


# In[ ]:


def rgb(img) :
    r = img[:,:,0:1] 
    g = img[:,:,1:2]
    b = img[:,:,2:3]
    return r,g,b

# In[ ]:


def uint8_RGB_to_float32(img):
    return img.astype(np.float32)/255.

def uint8_GRAY_to_float32(img):
    img = img.astype(np.float32)/255.
    img = np.expand_dims(img, axis=-1)
    # img = np.tile(img, (1, 1, 3))
    return img

def bool_to_float32(img):
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=-1)
    # img = np.tile(img, (1, 1, 3))
    return img

def float32_to_uint8(img):
    img = 255.*img
    return img.astype(np.uint8)

def to_float32(img):
    if img.dtype == 'uint8':
        if img.ndim == 2:
            return uint8_GRAY_to_float32(img)
        else:
            return uint8_RGB_to_float32(img)
    if img.dtype == 'bool':
        return bool_to_float32(img)


# In[ ]:


def rotate90(img) :
  dims = (img.shape[1], img.shape[0], img.shape[2])
  copie = np.zeros(shape=dims)
  for i in range(len(img[0])) :
    for j in range(len(img)) :
      copie[i][j] = img[j][i].copy()
  return copie



# In[ ]:
def reduce_resolution(img, k) :
  if k==0 :
    return img
  dims = (img.shape[0]//2, img.shape[1]//2, img.shape[2])
  copie = np.zeros(shape=dims)
  for i in range (copie.shape[0]):
    for j in range(copie.shape[1]) :
        copie[i,j]=1/4 * (img[2*i,2*j]+img[2*i+1,2*j]+img[2*i,2*j+1]+img[2*i+1,2*j+1])
  return reduce_resolution(copie,k-1)

# In[ ]:


def bilinear_interp(img, x, y) :
  if int(x)>=img.shape[0]-1 or int(x)<0 or int(y)<0 or int(y)>=img.shape[1]-1 :
    return np.zeros(shape = img.shape[2])
  x1 = int(np.floor(x))
  x2 = int(np.ceil(x))
  y1 = int(np.floor(y))
  y2 = int(np.ceil(y))
  dx = x-x1
  dy = y-y1
  Deltax = x2-x1
  Deltay = y2-y1
  Deltafx = img[x2,y1]-img[x1,y1]
  Deltafy = img[x1,y2]-img[x1,y1]
  Deltafxy = img[x1,y1] + img[x2,y2] - img[x2,y1]-img[x1,y2]
  if Deltax == 0 :
    if Deltay == 0 :
      return img[x1,y1]
    return(img[x1,y1]+Deltafy*dy/Deltay)
  if Deltay==0 :
    return(img[x1,y1]+Deltafx*dx/Deltax)
  return (Deltafx*dx/Deltax + Deltafy*dy/Deltay + Deltafxy*dx/Deltax*dy/Deltay+img[x1,y1])

def rescale(img,w_new, h_new) :
  dims = (h_new, w_new, img.shape[2])
  copie = np.zeros(shape=dims)
  for i in range (h_new) :
    for j in range (w_new) :
      copie[i][j] = bilinear_interp(img, i*img.shape[0]/h_new, j*img.shape[1]/w_new)
  return (copie)



def rotate(img, theta) :
  copie = np.zeros(shape=img.shape)
  for i in range (img.shape[0]) :
    for j in range (img.shape[1]) :
      x = img.shape[0]/2+np.cos(theta)*(i-img.shape[0]/2)-np.sin(theta)*(-j+img.shape[1]/2)
      y = img.shape[0]/2-np.sin(theta)*(i-img.shape[0]/2)-np.cos(theta)*(-j+img.shape[1]/2)
      copie[i,j] = bilinear_interp(img,x,y)
  return (copie)




# In[ ]:

def rgb_to_bandw(img) :
  if img.shape[2] == 1 :
    return img
  else :
    copie = np.zeros(shape=(img.shape[0],img.shape[1],1))
    for i in range(img.shape[0]) :
      for j in range(img.shape[1]) :
          copie[i,j]=0.21*img[i,j,0]+0.72*img[i,j,1]+0.07*img[i,j,2]
    return copie

# In[ ]:


filtre_gauss = (1./273.)*np.array([[1., 4. , 7. , 4. , 1.],
           [4., 16., 26., 16., 4.],
           [7., 26., 41., 26., 7.],
           [4., 16., 26., 16., 4.],
           [1., 4. , 7. , 4. , 1.]])


# In[ ]:


def convolution(img, w) :
  pad_copie=np.pad(img, pad_width = [(w.shape[0]//2,w.shape[0]//2),(w.shape[1]//2,w.shape[1]//2),(0,0)], mode = 'edge')
  res = np.zeros(shape = img.shape)
  for i in range (res.shape[0]) :
    for j in range (res.shape[1]) :
      for k in range(w.shape[0]) :
        for l in range(w.shape[1]) :
          res[i,j,0]+=w[k,l]*pad_copie[i+k,j+l,0]
  return res

def gaussian_blur(img) :
  return convolution(img, filtre_gauss)


# In[ ]:


w_y = np.array([[-1, 0, 1], 
                [-2, 0, 2], 
                [-1, 0, 1]], np.float32) 

#car y est ici notre axe horizontale

w_x = np.array([[ 1,  2, 1], 
                [ 0,  0, 0], 
                [-1, -2, -1]], np.float32)
#car x est ici notre axe verticale


def grad_x(img) :
  return convolution(img,w_x)

def grad_y(img) :
  return convolution(img,w_y)


# In[ ]:


def vis_grad(img) :
  gradx = grad_x(img)
  grady = grad_y(img)
  res_g = np.zeros(shape = (img.shape[0], img.shape[1], 1))
  res_b = np.zeros(shape = (img.shape[0], img.shape[1], 1))
  for i in range (img.shape[0]):
    for j in range (img.shape[1]):
      res_g[i,j,0] = gradx[i,j]
      res_b[i,j,0] = grady[i,j]
  return res_b, res_g


# In[ ]:


def polar_gradient(img) :
  copie = gaussian_blur(img)
  gradx = grad_x(copie)
  grady = grad_y(copie)

  # on normalise le gradient
  max = 0
  for i in range (copie.shape[0]) :
    for j in range (copie.shape[1]) :
      if max < (abs(gradx[i,j,0])+abs(grady[i,j,0])) :
        max = abs(gradx[i,j,0])+abs(grady[i,j,0])
  gradx = (1/max)*gradx
  grady = (1/max)*grady

  # on construit I et D
  I = abs(gradx) + abs(grady)
  D = np.arctan2(grady,gradx)

  return I,D


# In[ ]:


def non_max_suppression(I,D) :
  NMS = np.zeros(I.shape)
  for i in range(I.shape[0]) :
    for j in range(I.shape[1]) : #on regarde si le gradient est plus verticale ou plus horizontale,
                                 #puis on vérifie que notre case est plus grande que ses 2 voisins dans cette direction
      if ( (-3*np.pi/4<=D[i,j,0]<=-np.pi/4 or np.pi/4<=D[i,j,0]<=3*np.pi/4) and ( (j-1>0 and I[i,j-1,0]-I[i,j,0]>=0) or (j+1<I.shape[1] and I[i,j+1,0]-I[i,j,0]>=0)) ) or         ( (3*np.pi/4<=D[i,j,0] or D[i,j,0]<=-3*np.pi/4 or -np.pi/4<=D[i,j,0]<=np.pi/4) and ( (i-1>0 and I[i-1,j,0]-I[i,j,0]>=0) or (i+1<I.shape[0] and I[i+1,j,0]-I[i,j,0]>=0) ) ) :
        NMS[i,j,0] = 0
      else :
        NMS[i,j,0] = I[i,j,0]

  return NMS


# In[ ]:


def threshold(img, high, low) :
  copie = np.zeros(img.shape)
  for i in range(img.shape[0]) :
    for j in range(img.shape[1]) :
      if img[i,j,0]>=high :
        copie[i,j,0] = 1
      elif img[i,j,0]<=low :
        copie[i,j,0] = 0
      else :
        copie[i,j,0] = 0.3
  return copie


# In[ ]:


def hysteresis(img, weak=0.3) :
  copie = np.zeros(img.shape)
  for i in range(img.shape[0]) :
    for j in range(img.shape[1]) :
      if img[i,j,0]>0.99 :
        copie[i,j,0]=1
      elif weak-0.01<img[i,j,0]<weak+0.01:
        if (i-1>=0 and copie[i-1,j,0]>=0.99) or (j-1>=0 and copie[i,j-1,0]>0.99) or (i+1<img.shape[0] and copie[i+1,j,0]>0.99) or (j+1<img.shape[1] and copie[i,j+1,0]>0.99) or           (i-1>=0 and j-1>0 and copie[i-1,j-1,0]>=0.99) or (i+1<img.shape[0] and j-1>0 and copie[i+1,j-1,0]>=0.99) or           (i-1>=0 and j+1<img.shape[1] and copie[i-1,j+1,0]>=0.99) or (i+1<img.shape[0] and j+1<img.shape[1] and copie[i+1,j+1,0]>=0.99) :
          copie[i,j,0] = 1
        else :
          copie[i,j,0] = 0
      else :
        copie[i,j,0] = 0
  for i in range(img.shape[0]-1,-1,-1) : #comme on accepte les pixels indécis de manière dynamique, on le fait dans les 2 sens pour être sur d'en oublier aucun
    for j in range(img.shape[1]-1,-1,-1) :
      if img[i,j,0]>0.99 :
        copie[i,j,0]=1
      elif weak-0.01<img[i,j,0]<weak+0.01:
        if (i-1>=0 and copie[i-1,j,0]>=0.99) or (j-1>=0 and copie[i,j-1,0]>0.99) or (i+1<img.shape[0] and copie[i+1,j,0]>0.99) or (j+1<img.shape[1] and copie[i,j+1,0]>0.99) :
          copie[i,j,0] = 1
        else :
          copie[i,j,0] = 0
      else :
        copie[i,j,0] = 0
  return copie


# In[ ]:


def canny_edge_detection(img, high, low) :
  I,D = polar_gradient(img)
  NMS = non_max_suppression(I,D)
  res = hysteresis(threshold(NMS,high, low))
  return res


# In[ ]:


d2dx2 = np.array([[0, 0, 0], 
                  [1, -2, 1], 
                  [0, 0, 0]], np.float32) 

d2dxdy = np.array([[1, 0, -1], 
                   [0, 0, 0], 
                   [-1, 0, 1]], np.float32)

d2dy2 = np.array([[0, 1, 0], 
                  [0, -2, 0], 
                  [0, 1, 0]], np.float32)  


# In[ ]:


def vis_hessian(img) :
  H = np.zeros(shape = (img.shape[0], img.shape[1], 2, 2))
  v = np.zeros(shape = (img.shape[0], img.shape[1], 2))
  a = np.zeros(shape = (img.shape[0], img.shape[1]))
  s = np.zeros(shape = (img.shape[0], img.shape[1]))
  res = np.zeros(shape = (img.shape[0], img.shape[1], 2))
  res_g = np.zeros(shape = (img.shape[0], img.shape[1], 1))
  res_b = np.zeros(shape = (img.shape[0], img.shape[1], 1))
  d2imgdx2 = convolution(img, d2dx2)
  d2imgdxdy = convolution(img, d2dxdy)
  d2imgdy2 = convolution(img, d2dy2)
  res_g = np.zeros(shape = (img.shape[0], img.shape[1], 1))
  res_b = np.zeros(shape = (img.shape[0], img.shape[1], 1))
  for i in range(img.shape[0]) :
    for j in range(img.shape[1]) :
      H[i,j,0,0] = d2imgdx2[i,j,0]
      H[i,j,0,1] = d2imgdxdy[i,j,0]
      H[i,j,1,0] = d2imgdxdy[i,j,0]
      H[i,j,1,1] = d2imgdy2[i,j,0]
      val_propres, vect_propres = np.linalg.eigh(H[i,j])
      if abs(val_propres[0])>abs(val_propres[1]) :
        a[i,j] = np.abs(val_propres[0])
        v[i,j] = vect_propres[0]
      else :
        a[i,j] = np.abs(val_propres[1])
        v[i,j] = vect_propres[1]
      s[i, j] = np.abs(v[i, j, 1])
      H[i,j] = np.array([[val_propres[0], 0],[0, val_propres[1]]])
      res_g[i,j,] = a[i,j]*s[i,j]
      res_b[i,j,] = a[i,j]*(1-s[i,j])
      res[i,j] = np.array(a[i,j]*s[i,j],a[i,j]*(1-s[i,j]))

  return (res_g-np.min(res))/(np.max(res)-np.min(res)), (res_b-np.min(res))/(np.max(res)-np.min(res))

