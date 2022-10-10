from sklearn.neighbors import NearestCentroid
import torch
import math
from torch import batch_norm_elemt, batch_norm_gather_stats, nn


class PositionalEncoder(nn.Module):
    r"""
    Sine-cosine positional encoder for input points.
    """

    def __init__(self, d_input, n_freqs, log_space=False):
        super().__init__()

        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x: x]

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(
                2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

        # Alternate sin and cos
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(self, x):
        r"""
        Apply positional encoding to input.
        """
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)


class NeRF(nn.Module):
    r"""
    Neural radiance fields module.
    """

    def __init__(self, d_input=3, n_layers=8, d_filter=256, skip=(4,),log_space=False,n_freqs_views=4,n_freqs=10,near=2,far=6,nsamples=64,chunksize=4096,bq=True):
        super().__init__()

        self.d_input = d_input
        self.skip = skip
        self.act = nn.functional.relu
        self.chunksize = chunksize

        self.bq = bq

        self.near = near
        self.far = far
        self.nsamples = nsamples

          # Embedders
        self.encoder_viewdirs = PositionalEncoder(d_input, n_freqs_views,log_space=log_space)
        self.encoder = PositionalEncoder(d_input, n_freqs, log_space=log_space)

        self.d_viewdirs = self.encoder_viewdirs.d_output

        # Create model layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.encoder.d_output, d_filter)] +
            [nn.Linear(d_filter + self.encoder.d_output, d_filter) if i in skip
                else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
        )

        # Bottleneck layers
        self.alpha_out = nn.Linear(d_filter, 1)
        self.rgb_filters = nn.Linear(d_filter, d_filter)
        self.branch = nn.Linear(d_filter + self.d_viewdirs, d_filter // 2)
        self.output = nn.Linear(d_filter // 2, 3)


    def forward(self, x, viewdirs=None):
        r"""
        Forward pass with optional view direction.
        """
        # Apply forward pass up to bottleneck
        x_input = x
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x))
            if i in self.skip:
                x = torch.cat([x, x_input], dim=-1)

        # Apply bottleneck
        # Split alpha from network output
        alpha = self.alpha_out(x)

        # Pass through bottleneck to get RGB
        x = self.rgb_filters(x)
        x = torch.concat([x, viewdirs], dim=-1)
        x = self.act(self.branch(x))
        x = self.output(x)

        # Concatenate alphas to output
        x = torch.concat([x, alpha], dim=-1)
        return x

    def sample_stratified(self,rays_o,rays_d,near, far, n_samples, perturb=True):
        r"""
        Sample along ray from regularly-spaced bins.
        """

        # Grab samples for space integration along ray
        # t_vals = torch.linspace(0., 1., n_samples, device=rays_o.device)
        t_vals = torch.linspace(0., 1., n_samples, device=rays_o.device)
    
        # Sample linearly between `near` and `far`
        z_vals = near * (1.-t_vals) + far * (t_vals)
        
        # Draw uniform samples from bins along ray
        if perturb:
            mids = .5 * (z_vals[1:] + z_vals[:-1])
            upper = torch.concat([mids, z_vals[-1:]], dim=-1)
            lower = torch.concat([z_vals[:1], mids], dim=-1)
            t_vals = torch.rand([n_samples], device=z_vals.device)
            z_vals = lower + (upper - lower) * t_vals
        z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [n_samples])

        # Apply scale from `rays_d` and offset from `rays_o` to samples
        # pts: (width, height, n_samples, 3)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        return pts, z_vals


    def get_rays(self, height, width, focal_length, c2w):
        r"""
        Find origin and direction of rays through every pixel and camera origin.
        """
        # Apply pinhole camera model to gather directions at each pixel
        i, j = torch.meshgrid(
            torch.arange(width, dtype=torch.float32).to(c2w),
            torch.arange(height, dtype=torch.float32).to(c2w),
            indexing='ij')
        i, j = i.transpose(-1, -2), j.transpose(-1, -2)
        directions = torch.stack([(i - width * .5) / focal_length,
                                -(j - height * .5) / focal_length,
                                -torch.ones_like(i)
                                ], dim=-1)

        # Apply camera pose to directions
        rays_d = torch.sum(directions[..., None, :] * c2w[:3, :3], dim=-1)

        # Origin is same for all directions (the optical center)
        rays_o = c2w[:3, -1].expand(rays_d.shape)
        return rays_o, rays_d

    def get_chunks(self,inputs, chunksize: int = 2**15):
        r"""
        Divide an input into chunks.
        """
        return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

    def prepare_chunks(self,points, encoding_function, chunksize: int = 2**15):
        r"""
        Encode and chunkify points to prepare for NeRF model.
        """
        points = points.reshape((-1, 3))
        points = encoding_function(points)
        points = self.get_chunks(points, chunksize=chunksize)
        return points

    def prepare_viewdirs_chunks(self,points, rays_d, encoding_function, chunksize=2**15):
        r"""
        Encode and chunkify viewdirs to prepare for NeRF model.
        """
        # Prepare the viewdirs
        viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        viewdirs = viewdirs[:, None, ...].expand(points.shape).reshape((-1, 3))
        viewdirs = encoding_function(viewdirs)
        viewdirs = self.get_chunks(viewdirs, chunksize=chunksize)
        return viewdirs

    def render(self, height, width, focal_length, c2w):

        rays_o, rays_d = self.get_rays(height, width, focal_length, c2w)
        rays_o = rays_o.reshape([-1, 3])
        rays_d = rays_d.reshape([-1, 3])

        query_points, z_vals = self.sample_stratified(
            rays_o, rays_d, self.near, self.far,self.nsamples,perturb=True)

        batches = self.prepare_chunks(query_points, self.encoder, chunksize=self.chunksize)
        batches_viewdirs = self.prepare_viewdirs_chunks(query_points, rays_d, self.encoder_viewdirs, chunksize=self.chunksize)

        predictions = []
        for batch, batch_viewdirs in zip(batches, batches_viewdirs):
            predictions.append(self.forward(batch, viewdirs=batch_viewdirs))
        raw = torch.cat(predictions, dim=0)
        raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

        # Perform differentiable volume rendering to re-synthesize the RGB image.
        if self.training:
            rgb_map, depth_map, acc_map, weights, uncertainty = self.raw2outputs(raw, z_vals, rays_d,raw_noise_std=0.01)
        else:
            rgb_map, depth_map, acc_map, weights, uncertainty = self.raw2outputs(raw, z_vals, rays_d,raw_noise_std=0.0)

        return rgb_map, depth_map, acc_map, weights, uncertainty

    def cumprod_exclusive(self, tensor):
        r"""
        (Courtesy of https://github.com/krrish94/nerf-pytorch)

        Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

        Args:
        tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
            is to be computed.
        Returns:
        cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
            tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
        """

        # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
        cumprod = torch.cumprod(tensor, -1)
        # "Roll" the elements along dimension 'dim' by 1 element.
        cumprod = torch.roll(cumprod, 1, -1)
        # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
        cumprod[..., 0] = 1.

        return cumprod

    def matern(self,x,y,rho=1/5):

        return (1. + math.sqrt(3.)*torch.abs(x-y)/rho)*torch.exp(-math.sqrt(3.)*torch.abs(x-y)/rho)

    def bayes_quad_mu(self,x,rho=1/5):

        y = 4.*rho/math.sqrt(3.) - 1./3.*torch.exp(math.sqrt(3.)*(x-1.)/rho)*(3.+2.*math.sqrt(3.)*rho -3.*x) - 1./3.*torch.exp(-math.sqrt(3.)/rho*x)*(3.*x + 2.*math.sqrt(3.)*rho)

        return y  

    def bayes_quad_sig(self,rho=1/5):
    
        y = 2*rho/3*(2*math.sqrt(3) - 3*rho + math.exp(-math.sqrt(3)/rho)*(math.sqrt(3) + 3*rho))
    
        return y

    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0.01):
        r"""
        Convert the raw NeRF output into RGB and other maps.
        """

        # Difference between consecutive elements of `z_vals`. [n_rays, n_samples]
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat(
            [dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        # Add noise to model's predictions for density. Can be used to
        # regularize network during training (prevents floater artifacts).
        noise = 0.
        if raw_noise_std > 0.:
            noise = (torch.randn(raw[..., 3].shape) * raw_noise_std).to(raw.device)

        rgb = torch.sigmoid(raw[..., :3])  # [n_rays, n_samples, 3]

        # Bayesian Quadrature rendering
        if self.bq:

            t_vals = (z_vals[0,:]-self.near)/(self.far-self.near)
            weights = nn.functional.relu(raw[..., 3] + noise)
            kxx = self.matern(t_vals.reshape(-1,1),t_vals.reshape(1,-1))
            
            bqm = self.bayes_quad_mu(t_vals.reshape(-1,1)).reshape(1,-1)
            bqs = self.bayes_quad_sig() - bqm@torch.linalg.solve(kxx,bqm.T)
            # pre_mult = bqm@torch.linalg.inv(kxx)

            # rgb_map = (pre_mult.repeat(rgb.shape[0],1,1)@(weights.unsqueeze(-1)*rgb).squeeze()).squeeze()

            rgb_map = (bqm@torch.linalg.solve(kxx,weights.unsqueeze(-1)*rgb)).squeeze()
            depth_map = torch.sum(weights*z_vals,-1)
            acc_map = torch.sum(weights, dim=-1)

        else:

            # Predict density of each sample along each ray. Higher values imply
            # higher likelihood of being absorbed at this point. [n_rays, n_samples]
            alpha = 1.0 - torch.exp(-nn.functional.relu(raw[..., 3] + noise) * dists)

            # Compute weight for RGB of each sample along each ray. [n_rays, n_samples]
            # The higher the alpha, the lower subsequent weights are driven.
            weights = alpha*self.cumprod_exclusive(1. - alpha + 1e-10)

            # Compute weighted RGB map.
            rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [n_rays, 3]

            # Estimated depth map is predicted distance.
            depth_map = torch.sum(weights * z_vals, dim=-1)

            # Disparity map is inverse depth.
            disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map),
                                    depth_map / torch.sum(weights, -1))

            # Sum of weights along each ray. In [0, 1] up to numerical error.
            acc_map = torch.sum(weights, dim=-1)

            bqs = torch.ones(1).to(alpha.device)

        return rgb_map, depth_map, acc_map, weights, torch.nn.functional.relu(bqs)
