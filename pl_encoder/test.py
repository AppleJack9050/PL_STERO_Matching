class PointTransformer(nn.Module):
    def __init__(self, trans_dim=256, depth=12, drop_path_rate=0.1, num_heads=8, group_size=32, num_group=784, encoder_dims=256):
        super().__init__()
        # self.config = config

        self.trans_dim = trans_dim
        self.depth = depth
        self.drop_path_rate = drop_path_rate 
        # self.cls_dim = config.cls_dim 
        self.num_heads = num_heads 

        self.group_size = group_size
        self.num_group = num_group

        # fine level of group
        self.fine_group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        # coarse level of group
        self.fine_group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        # define the encoder
        self.encoder_dims = encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims,self.trans_dim)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        # self.cls_head_finetune = nn.Sequential(
        #     nn.Linear(self.trans_dim * 2, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, self.cls_dim)
        # )

        # self.build_loss_func()
        
    def forward(self, pts):
        # divide the point cloud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  #  B G N
        # group_input_tokens = self.reduce_dim(group_input_tokens)

        # prepare cls
        # cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
        # cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  

        # add pos embedding
        pos = self.pos_embed(center)

        # final input
        # x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        # pos = torch.cat((cls_pos, pos), dim=1)
        # transformer


        x = self.blocks(group_input_tokens, pos)
        x = self.norm(x)

        # concat_f = torch.cat([x[:,0], x[:, 1:].max(1)[0]], dim = -1)
        # ret = self.cls_head_finetune(concat_f)
        return x

"""
This program could be accelaerated by cuda
"""

def fps(xyz, npoint):
    """
    Furthest Point Sampling
    Input:
        xyz: torch.Tensor of shape (B, N, 3) representing the batch of point clouds.
        npoint: int, number of points to sample (G).
    Output:
        new_xyz: torch.Tensor of shape (B, npoint, 3) representing the sampled center points.
    """
    device = xyz.device
    B, N, _ = xyz.shape
    new_xyz = torch.zeros(B, npoint, 3, device=device)  # To store sampled points
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)  # To store indices of the points
    # Initialize distances as infinity
    distances = torch.full((B, N), float('inf'), device=device)
    
    # Randomly choose the first point for each batch
    farthest = torch.randint(0, N, (B,), device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        new_xyz[:, i, :] = xyz[batch_indices, farthest, :]
        # Expand the selected point to subtract from all points: shape becomes (B, 1, 3)
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)
        # Compute squared distances from the current centroid to all points
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        # Update distances (keep the minimum distance so far for each point)
        mask = dist < distances
        distances[mask] = dist[mask]
        # Select the next point as the one with the maximum distance from the sampled set
        farthest = torch.max(distances, dim=1)[1]
    
    return new_xyz

class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        # self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center) # B G M
        idx = knn_point(self.group_size, xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)
    
def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist