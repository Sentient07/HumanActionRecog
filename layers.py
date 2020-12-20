# embeddings.py


from __future__ import division
import torch.nn as nn

import torch
import torch.nn.functional as F



class AppearancePrecomp(nn.Module):
    def __init__(self, embed_size, name, d_appearance=1024, d_hidden=1024, normalize=False, dropout=True, num_layers=2):
        super(AppearancePrecomp, self).__init__()
        print("Hidden size is " + str(d_hidden))
        self.name = name
        self.num_layers = num_layers
        self.normalize = normalize

        # Projection layer
        if self.num_layers==2:
            self.proj = Proj_2Unit(d_appearance, embed_size, hidden_dimension=d_hidden, normalize=normalize, dropout=dropout)
        elif self.num_layers==1:
            self.proj = Proj_1Unit(d_appearance, embed_size, normalize=normalize)


    def forward(self, batch_input):

        if self.name=='subject':
            output = batch_input['precompappearance'][:,0,:]
        elif self.name=='object':
            output = batch_input['precompappearance'][:,1,:]
        elif self.name=='word':
            output = batch_input['precompappearance']

        if self.num_layers>0:
            output = self.proj(output) # normalization is included
        else:
            if self.normalize:
                output = F.normalize(output,2,-1) # L2 norm according to last dimension

        return output



class SpatialAppearancePrecomp(nn.Module):
    def __init__(self, embed_size, d_appearance=1024, d_hidden=1024, normalize=False, dropout=True, num_layers=2):
        super(SpatialAppearancePrecomp, self).__init__()
        print("Hidden size is " + str(d_hidden))

        self.normalize = normalize
        self.num_layers = num_layers

        # 2 fc layer + normalize
        self.spatial_module = SpatialRawCrop(400, normalize=True)

        # Project appearance feats in subspace 300 (mimic PCA iccv17) before concatenating with spatial
        self.appearance_module = nn.Linear(d_appearance, 300)

        # Aggregate spatial and appearance feature with fc layer
        if self.num_layers==2:
            self.proj = Proj_2Unit(400+600, embed_size, hidden_dimension=d_hidden, normalize=normalize, dropout=dropout)
        elif self.num_layers==1:
            self.proj = Proj_1Unit(400+600, embed_size, normalize=normalize)


    def forward(self, batch_input):

        # Spatial feats
        spatial_feats = self.spatial_module(batch_input) # already L2 norm

        # Appearance feats subject L2 norm 
        appearance_human = batch_input['precompappearance'][:,0,:]
        appearance_human = F.normalize(self.appearance_module(appearance_human))

        # Appearance feats object L2 norm
        appearance_object = batch_input['precompappearance'][:,1,:]
        appearance_object = F.normalize(self.appearance_module(appearance_object))

        # Concat both L2 norm
        appearance_feats = torch.cat([appearance_human, appearance_object],1)
        appearance_feats = F.normalize(appearance_feats)

        # Concat appearance and spatial
        output = torch.cat([spatial_feats, appearance_feats],1)
        # Proj
        if self.num_layers > 0:
            output = self.proj(output)
     
        else: 
            if self.normalize:
                output = F.normalize(output,2,-1) # L2 norm according to last dimension
                #output = F.normalize(output) #old code: check this was doing the same

        return output



class SpatialRawCrop(nn.Module):
    """
    Baseline model using only spatial coordinates of subject and object boxes,
    # i.e renormalized [x1, y1, w1, h1, x2, y2, w2, h2] in the coordinates of union boxes
    """
    def __init__(self, embed_size, normalize=False):
        super(SpatialRawCrop, self).__init__()

        self.embed_size = embed_size
        self.normalize = normalize

        self.raw_coordinates = CroppedBoxCoordinates()

        self.net = nn.Sequential(nn.Linear(8, 128),
                                nn.ReLU(),
                                nn.Linear(128,self.embed_size),
                                nn.ReLU())


    def forward(self, batch_input):

        pair_objects = batch_input['pair_objects']
        output = self.raw_coordinates(pair_objects)
        output = self.net(output)

        if self.normalize:
            output = F.normalize(output)

        return output


class LanguageProj(nn.Module):
    """
    Projection of query embeddings (N,num_words,P) into (N,embed_size)
    If preceeding layer is word2vec P=300 and the query embeddings are word2vec features
    If preceeding layer is onehot P=vocab_size and the query embeddings are onehot vectors
    Different aggregation functions
    input_dim is usually 300 (word2vec dim) but it can be also the size of vocabulary (if working with 1-hot encoding) 
    """
    def __init__(self, embed_size, num_words=1, input_dim=300, num_layers=2, hidden_dimension=None,
    			 gated_unit=False, normalize=True, dropout=False, aggreg='concatenation'):
        super(LanguageProj, self).__init__()
 
        self.num_words = num_words 
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.aggreg = aggreg # How to aggregate across word if num_words>1       

        # Compute dimension of input vector given to proj (this depends on how you aggregate the words)
        if self.aggreg=='concatenation':
            input_dim_words = self.input_dim*self.num_words
        elif self.aggreg=='average':
            input_dim_words = self.input_dim
 
        # Projection layer
        if gated_unit:
            self.proj = Gated_Embedding_Unit(input_dim*self.num_words, self.embed_size) # already normalized by default
        else:
            if num_layers==2:
                if not hidden_dimension:
                    hidden_dimension=input_dim
                self.proj = Proj_2Unit(input_dim_words, self.embed_size, hidden_dimension, normalize, dropout)
            elif num_layers==1:
                # Only 1 layer
                self.proj = Proj_1Unit(input_dim_words, self.embed_size, normalize)


    def forward(self, batch_embeddings):

        if self.aggreg=='concatenation':
            feats = torch.cat([batch_embeddings[:,j,:] for j in range(self.num_words)],1) # Concatenate embeddings to form Nx(300*num_words)
        elif self.aggreg=='average':
            feats = torch.mean(batch_embeddings,1)

        if self.num_layers>0:
            feats = self.proj(feats)

        return feats


class CroppedBoxCoordinates(nn.Module):
    def __init__(self):
        super(CroppedBoxCoordinates, self).__init__()

    def forward(self, pair_objects):

        """
        Get cropped box coordinates [x1,y1,w1,h1,x2,y2,w2,h2] as if the image was the union of boxes and renormalize by union of box area
        """

        subject_boxes = pair_objects[:,0,:4]
        object_boxes = pair_objects[:,1,:4]
        union_boxes = torch.cat((torch.min(subject_boxes[:,:2], object_boxes[:,:2]), torch.max(subject_boxes[:,2:4], object_boxes[:,2:4])),1)
        width_union = union_boxes[:,2]-union_boxes[:,0]+1
        height_union = union_boxes[:,3]-union_boxes[:,1]+1
        area_union = width_union*height_union
        area_union = area_union.sqrt()
        area_union = area_union.unsqueeze(1).expand_as(subject_boxes)

        # Get x,y,w,h in union box coordinates system: copy input
        subject_boxes_trans = subject_boxes.clone()
        object_boxes_trans = object_boxes.clone()
        subject_boxes_trans[:,0].data.copy_(subject_boxes_trans.data[:,0]-union_boxes.data[:,0])
        subject_boxes_trans[:,2].data.copy_(subject_boxes_trans.data[:,2]-union_boxes.data[:,0])
        subject_boxes_trans[:,1].data.copy_(subject_boxes_trans.data[:,1]-union_boxes.data[:,1])
        subject_boxes_trans[:,3].data.copy_(subject_boxes_trans.data[:,3]-union_boxes.data[:,1])
        object_boxes_trans[:,0].data.copy_(object_boxes_trans.data[:,0]-union_boxes.data[:,0])
        object_boxes_trans[:,2].data.copy_(object_boxes_trans.data[:,2]-union_boxes.data[:,0])
        object_boxes_trans[:,1].data.copy_(object_boxes_trans.data[:,1]-union_boxes.data[:,1])
        object_boxes_trans[:,3].data.copy_(object_boxes_trans.data[:,3]-union_boxes.data[:,1])

        # Get x,y,w,h
        subject_boxes_trans[:,2:4].data.copy_(subject_boxes_trans.data[:,2:4]-subject_boxes_trans.data[:,:2]+1)
        object_boxes_trans[:,2:4].data.copy_(object_boxes_trans.data[:,2:4]-object_boxes_trans.data[:,:2]+1)
        

        # Renormalize by union box size
        subject_boxes_trans = subject_boxes_trans.mul(1/area_union)
        object_boxes_trans = object_boxes_trans.mul(1/area_union)

        output = torch.cat([subject_boxes_trans, object_boxes_trans],1)

        return output


class View(nn.Module):
    def __init__(self, outdim):
        super(View, self).__init__()
        self.outdim = outdim

    def forward(self, x):
        return x.view(-1, self.outdim)


class Proj_2Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension, hidden_dimension=None, normalize=False, dropout=True):
        super(Proj_2Unit, self).__init__()

        if not hidden_dimension:
            hidden_dimension = input_dimension
        self.normalize = normalize

        if dropout:
            self.proj = nn.Sequential(nn.Linear(input_dimension, hidden_dimension),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.Linear(hidden_dimension, output_dimension))

        else:

            self.proj = nn.Sequential(nn.Linear(input_dimension, hidden_dimension),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dimension, output_dimension))


    def forward(self,x):

        x = self.proj(x)

        if self.normalize:
            x = F.normalize(x,2,-1) # L2 norm according to last dimension

        return x


class Proj_1Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension, normalize=False):
        super(Proj_1Unit, self).__init__()

        self.normalize = normalize
        self.proj = nn.Linear(input_dimension, output_dimension)

    def forward(self,x):

        x = self.proj(x)

        if self.normalize:
            x = F.normalize(x,2,-1) # L2 norm according to last dimension

        return x


class Gated_Embedding_Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(Gated_Embedding_Unit, self).__init__()

        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = Context_Gating(output_dimension)
  
    def forward(self,x):
        
        x = self.fc(x)
        x = self.cg(x)
        x = F.normalize(x,2,-1)

        return x


class Context_Gating(nn.Module):
    """ Taken from Miech et al. """
    def __init__(self, dimension, add_batch_norm=True):
        super(Context_Gating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(dimension)
        
    def forward(self,x):
        x1 = self.fc(x)

        if self.add_batch_norm:
            x1 = self.batch_norm(x1) 

        x = torch.cat((x, x1), 1)
        
        return F.glu(x,1)