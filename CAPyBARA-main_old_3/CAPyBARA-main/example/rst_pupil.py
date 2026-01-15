import numpy as np
import matplotlib.pyplot as plt
import os


def _make_spider_positions(D, c, is_lyot=False):
    """
    Calculate spider positions for the telescope aperture.

    Args:
        D (float): Diameter of the Roman Telescope
        c (float): Scaling constant
        is_lyot (bool): Whether this is for a Lyot mask

    Returns:
        dict: Dictionary containing spider geometry parameters
    """
    if is_lyot == False:
        return {
            'width': 0.65 * D / c,
            'width2': 1.5 * D / c,
            'offset': 2.45 * D / c,
            'outer_radius': 10 * D / c,
            'outer_radius2': 3.25 * D / c,
            'D': D
        }
    else:
        return {
            'width': 0.7 * D / c,
            'width2': 1.5 * D / c,
            'offset': 2.45 * D / c,
            'outer_radius': 10 * D / c,
            'outer_radius2': 3.25 * D / c,
            'D': D
        }


def _normalize_positions(positions):
    """Normalize the positions based on the diameter."""
    for key in positions:
        if key != 'D':
            positions[key] /= positions['D']
    positions['D'] = 1.0


def make_spider(start, end, width):
    """
    Create a spider (strut) mask function.
    
    Args:
        start (array): Starting position [x, y]
        end (array): Ending position [x, y]
        width (float): Width of the spider
    
    Returns:
        function: Function that takes (x, y) coordinates and returns mask
    """
    def spider_mask(x, y):
        # Vector from start to end
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)
        
        # Normalize direction vector
        dx_norm = dx / length
        dy_norm = dy / length
        
        # Vector from start to each point
        px = x - start[0]
        py = y - start[1]
        
        # Project onto spider direction
        projection = px * dx_norm + py * dy_norm
        
        # Perpendicular distance from spider centerline
        perp_dist = np.abs(px * dy_norm - py * dx_norm)
        
        # Mask is 0 (blocked) where:
        # 1. Point is within spider width
        # 2. Point is between start and end along spider direction
        mask = np.ones_like(x)
        in_width = perp_dist <= width / 2
        in_length = (projection >= 0) & (projection <= length)
        mask[in_width & in_length] = 0
        
        return mask
    
    return spider_mask


def make_obstructed_circular_aperture(diameter, obscuration_ratio):
    """
    Create an obstructed circular aperture function.
    
    Args:
        diameter (float): Outer diameter of aperture
        obscuration_ratio (float): Ratio of central obscuration to outer diameter
    
    Returns:
        function: Function that takes (x, y) coordinates and returns mask
    """
    def aperture_mask(x, y):
        r = np.sqrt(x**2 + y**2)
        outer_radius = diameter / 2
        inner_radius = outer_radius * obscuration_ratio
        
        mask = np.zeros_like(x)
        mask[(r <= outer_radius) & (r >= inner_radius)] = 1
        
        return mask
    
    return aperture_mask


def _create_spiders(positions, with_spiders=True, rotation_angle=0):
    """
    Create the spider structures with optional rotation.
    
    This function generates the spider (strut) obstruction pattern for a telescope aperture.
    The spiders can be rotated by applying a coordinate transformation to the input grid.
    
    Args:
        positions (dict): Dictionary containing spider geometry parameters
        with_spiders (bool): Whether to include spiders. if False, returns a function that always returns 1 (no obstruction). It was added to facilitate comparison between apertures with and without spiders.
        rotation_angle (float): Rotation angle in degrees. Default is 0 (no rotation).

    Returns:
        function: Function that takes (x, y) coordinates and returns spider mask

    Note: 
        The rotation is applied to the coordinate grid before evaluating the spider masks. We chose this approach to avoid the complexity of rotating each spider individually.

        Hence we apply the inverse rotation to the coordinates when evaluating the combined spider mask by rotating the coordinates by -rotation_angle. So that the spiders appear rotated by +rotation_angle in the final mask.
    
    """
    if not with_spiders:
        return lambda x, y: np.ones_like(x)

    spider_width, spider_width2 = positions['width'], positions['width2']
    spider_inner_radius, spider_outer_radius, spider_outer_radius2 = (
        positions['offset'], positions['outer_radius'], positions['outer_radius2']
    )
    pi = np.pi

    spider_configs = [
        (pi / 6, 8 * pi / 180, spider_width, spider_outer_radius, False, False, False, True),
        (pi / 6, pi / 6 + 37.71 * pi / 180, spider_width, spider_outer_radius, False, False, False, False),
        (5 * pi / 6, pi / 6 + 37.71 * pi / 180, spider_width, spider_outer_radius, False, False, True, False),
        (5 * pi / 6, 8 * pi / 180, spider_width, spider_outer_radius, False, False, True, True),
        (3 * pi / 2, 3 * pi / 2 - 8 * pi / 180 - pi / 6, spider_width, spider_outer_radius, False, False, False, False),
        (3 * pi / 2, 3 * pi / 2 + 8 * pi / 180 + pi / 6, spider_width, spider_outer_radius, False, False, False, False),
        (pi / 6, pi / 6, spider_width2, spider_outer_radius2, False, False, False, False),
        (5 * pi / 6, pi / 6, spider_width2, spider_outer_radius2, False, False, True, False),
        (3 * pi / 2, 3 * pi / 2, spider_width2, spider_outer_radius2, False, False, False, False)
    ]

    spiders = []
    for start_angle, end_angle, width, radius, invert_cos_start, invert_sin_start, invert_cos_end, invert_sin_end in spider_configs:
        spider_start = spider_inner_radius * np.array([
            np.cos(start_angle) if not invert_cos_start else -np.cos(start_angle),
            np.sin(start_angle) if not invert_sin_start else -np.sin(start_angle)
        ])
        spider_end = radius * np.array([
            np.cos(end_angle) if not invert_cos_end else -np.cos(end_angle),
            np.sin(end_angle) if not invert_sin_end else -np.sin(end_angle)
        ])
        spiders.append(make_spider(spider_start, spider_end, width))

    def _combined_spiders_base(x, y):
        product = np.ones_like(x)
        for spider in spiders:
            product *= spider(x, y)
        return product

    if rotation_angle != 0:
        # Apply inverse rotation to coordinates (rotate grid clockwise to make spiders appear counterclockwise)
        angle = np.deg2rad(rotation_angle)
        cos_rot = np.cos(-angle)
        sin_rot = np.sin(+angle)

        def _combined_spiders(x, y):
            # Apply 2D rotation matrix for angle -θ
            x_rot = cos_rot * x - sin_rot * y
            y_rot = sin_rot * x + cos_rot * y
            return _combined_spiders_base(x_rot, y_rot)
        
        return _combined_spiders
    else:
        return _combined_spiders_base

def _make_obstructed_aperture_with_spiders(D, central_obscuration_ratio, with_spiders=True, normalized=True, is_lyot=False, spider_rotation_angle=0):
    """Create an obstructed aperture with optional spider structures."""
    c = 19.6
    positions = _make_spider_positions(D, c, is_lyot)

    if normalized:
        _normalize_positions(positions)
        D = 1.0

    if is_lyot is True:
        D *= 0.8

    obstructed_aperture = make_obstructed_circular_aperture(D, central_obscuration_ratio)
    spider_structure = _create_spiders(positions, with_spiders, spider_rotation_angle)

    def aperture_function(x, y):
        """
        Evaluate aperture at coordinates (x, y).
        
        Args:
            x (ndarray): X coordinates
            y (ndarray): Y coordinates
        
        Returns:
            ndarray: Aperture mask values
        """
        return obstructed_aperture(x, y) * spider_structure(x, y)

    return aperture_function


def _make_rst_aperture(normalized=True, with_spiders=True, spider_rotation_angle=0):
    """Create the RST aperture function."""
    return _make_obstructed_aperture_with_spiders(2.36, 0.303, with_spiders, normalized, is_lyot=False, spider_rotation_angle=spider_rotation_angle)


def _make_lyot_mask(normalized=True, with_spiders=True, spider_rotation_angle=0):
    """Create the Lyot mask function."""
    return _make_obstructed_aperture_with_spiders(2.36, 0.6, with_spiders, normalized, is_lyot=True, spider_rotation_angle=spider_rotation_angle)

# Example usage functions
def create_coordinate_grid(size=128, diameter=1.0, padding=1):
    """
    Create a coordinate grid for evaluating aperture functions.
    
    Args:
        size (int): Number of pixels along each dimension
        diameter (float): Diameter of the pupil (default: 1.0 for normalized apertures)
        padding (int): Padding factor - array extent will be padding times the pupil radius
                      (e.g., padding=2 means array extends from -D to +D, or 2x the pupil diameter)
    
    Returns:
        tuple: (x, y) coordinate arrays
    
    Examples:
        # Default: array from -0.5 to +0.5 (covers diameter of 1.0)
        x, y = create_coordinate_grid(size=128, diameter=1.0, padding=1)
        
        # With padding=2: array from -1.0 to +1.0 (2x the pupil diameter)
        x, y = create_coordinate_grid(size=128, diameter=1.0, padding=2)
    """
    total_size = int(size * padding)
    pixel_scale = diameter / size # how many pixels across the diameter
    extent = (total_size / 2) * pixel_scale
    coords = np.linspace(-extent, extent, total_size)
    x, y = np.meshgrid(coords, coords)
    return x, y


def visualize_aperture(aperture_func, size=128, diameter=1.0, padding=1, title="Aperture"):
    """
    Visualize an aperture function.
    
    Args:
        aperture_func: Function that takes (x, y) and returns mask
        size (int): Grid size
        diameter (float): Diameter of the pupil
        padding (int): Padding factor for the array size
        title (str): Plot title
    """
    x, y = create_coordinate_grid(size, diameter, padding)
    aperture = aperture_func(x, y)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(aperture, 
               origin='lower', cmap='gray',
               extent=[-diameter/2*padding, diameter/2*padding, 
                      -diameter/2*padding, diameter/2*padding])
    plt.colorbar(label='Transmission')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example: Create and visualize RST aperture with default padding
    rst_aperture = _make_rst_aperture(normalized=False, with_spiders=True)
    visualize_aperture(rst_aperture, diameter=2.36, padding=1, 
                      title="RST Aperture with Spiders (padding=1)")
    
    # Example: Create and visualize RST aperture with padding=2
    visualize_aperture(rst_aperture, diameter=2.36, padding=2,
                      title="RST Aperture with Spiders (padding=2)")
    
    # Example: Create and visualize Lyot mask
    lyot_mask = _make_lyot_mask(normalized=True, with_spiders=True)
    visualize_aperture(lyot_mask, diameter=0.8, padding=1.5,
                      title="Lyot Mask with Spiders (padding=1.5)")

    # Try multiple rotation angles
    rotation_angles = [0, 15, 30, 45]  # 0°, 15°, 30°, 45°
    rotation_labels = ['0°', '15°', '30°', '45°']
    
    size = 256
    x, y = create_coordinate_grid(size=size, diameter=2.36, padding=2)
    extent = [-1, 1, -1, 1]
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for i, (angle, label) in enumerate(zip(rotation_angles, rotation_labels)):
        rst_aperture = _make_rst_aperture(normalized=False, with_spiders=True, spider_rotation_angle=angle)
        aperture = rst_aperture(x, y)
        
        im = axes[i].imshow(aperture, origin='lower', cmap='gray')
        axes[i].set_title(f'Rotation: {label}', fontsize=12)
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        plt.colorbar(im, ax=axes[i], label='Transmission', fraction=0.046)
    
    plt.tight_layout()
    plt.show()