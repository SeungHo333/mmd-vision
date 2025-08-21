import torch
import numpy as np
from PIL import Image, ImageDraw

# Imports for the vision model and utilities
from langsam.lang_sam import LangSAM
from langsam.lang_sam.utils import draw_image

# Imports from the planning codebase
from scripts.inference.vision_inference_multi_agent import run_inference

device='cuda' if torch.cuda.is_available() else 'cpu'

def goal_list_generator(goals, image_size):
    """
    Converts object detection results (box corners) into normalized goal
    coordinates for the motion planner.
    
    Args:
        goals (dict): Dictionary containing 'boxes' (np.ndarray of box corners)
                      and 'labels' from the vision model.
        image_size (tuple): The width and height of the original image.
        
    Returns:
        list[torch.Tensor]: A list of 2D tensors, where each tensor
                             represents a goal's (x, y) coordinates.
    """
    box_corners = goals["boxes"]
    goal_labels = goals["labels"]
    scale = 2 / image_size[0]
    goal_list = []

    for i in range(box_corners.shape[0]):
        # Calculate the center of the bounding box
        x_avg = (box_corners[i, 0] + box_corners[i, 2]) / 2
        y_avg = (box_corners[i, 1] + box_corners[i, 3]) / 2

        # Normalize coordinates to the planner's [-1, 1] range.
        # The y-axis is inverted for the conversion.
        x_scaled = x_avg * scale - 1
        y_scaled = y_avg * -scale + 1

        goal_tensor = torch.tensor((x_scaled, y_scaled), dtype=torch.float32, device=device)
        goal_list.append(goal_tensor)

        print(f"Goal '{goal_labels[i]}': Raw coords ({x_avg:.2f}, {y_avg:.2f}) -> "
              f"Normalized coords ({x_scaled:.2f}, {y_scaled:.2f})")
    
    return goal_list

def detect_goals_from_image(image_pil=None):
    """
    Detects goals in an image using the LangSAM model and a text prompt.
    
    Args:
        image_pil (Image.Image): The input image to process.
        
    Returns:
        list[torch.Tensor]: A list of normalized torch tensors for goal positions.
    """
    # Task image size
    image_size = image_pil.size

    # Set the text prompt for the vision model
    text_prompt = "maple leaf. basketball. fire hydrant. blue cup. soccer ball. light bulb."

    # Create the vision model instance
    model = LangSAM()

    # Run prediction
    results = model.predict([image_pil], [text_prompt])

    # box_corners = results[0]["boxes"]
    goals = results[0]
    goal_list = goal_list_generator(goals, image_size)
    
    output_image = draw_image(
        np.asarray(image_pil),
        results[0]["masks"],
        results[0]["boxes"],
        results[0]["scores"],
        results[0]["labels"],
    )

    output_path = "tasks/drop_region/results/detected_object.png"
    Image.fromarray(np.uint8(output_image)).save(output_path)
    print(f"Saved image with detected objects to {output_path}")

    return goal_list

def plot_trajectories_on_image(
    trajectories, 
    background_img, 
    output_path,
    line_width=3,
    state_dot_size=3,
    start_dot_size=12,
    end_dot_size=12
):
    """
    Plots all agent trajectories on a background image.
    
    Args:
        trajectories (list[torch.Tensor]): A list of agent trajectories.
        background_img (PIL.Image): The background image.
        output_path (str): The file path to save the final image.
        line_width (int): The width of the path lines.
        state_dot_size (int): The size of the dots for each state.
        start_dot_size (int): The size of the start point dot.
        end_dot_size (int): The size of the end point dot.
    """
    mapp = background_img.copy()
    draw = ImageDraw.Draw(mapp)
    
    num_agents = len(trajectories)
    img_height, img_width = mapp.size

    agent_colors = ["red", "green", "blue", "orange", "purple", "cyan", "magenta"]
    agent_trajectories = trajectories

    for i in range(num_agents):
        color = agent_colors[i % len(agent_colors)]
        
        path_points = agent_trajectories[i].cpu().numpy()
        path_points = path_points[:,:2] # x / y

        # Convert normalized [-1, 1] coordinates to pixel coordinates [0, img_size].
        # The y-axis is inverted.
        h_coords = (path_points[:, 0] - 1) / 2 * -img_height
        w_coords = (path_points[:, 1] + 1) / 2 * img_width

        pixel_path = list(zip(w_coords, h_coords))
        
        # 1. Draw the entire path as a line
        if len(pixel_path) > 1:
            draw.line(pixel_path, fill=color, width=line_width)
        
        # 2. Draw a circle for each state along the path
        for point in pixel_path:
            px, py = point
            draw.ellipse(
                [(px - state_dot_size/2, py - state_dot_size/2),
                 (px + state_dot_size/2, py + state_dot_size/2)],
                fill=color
            )
        
        # 3. Draw the start and end points again to highlight them
        start_x, start_y = pixel_path[0]
        end_x, end_y = pixel_path[-1]
        
        # End point (goal) - white dot
        draw.ellipse(
            [(start_x - start_dot_size/2, start_y - start_dot_size/2),
             (start_x + start_dot_size/2, start_y + start_dot_size/2)],
            fill="white", outline=color, width=2
        )

        # Start point - black dot
        draw.ellipse(
            [(end_x - end_dot_size/2, end_y - end_dot_size/2),
             (end_x + end_dot_size/2, end_y + end_dot_size/2)],
            fill=color, outline="black", width=2
        )
        
    mapp.save(output_path)

if __name__ == '__main__':
    # --- Main Execution Flow ---

    # 1. Load the input task image
    image_path = "tasks/drop_region/top_down_view_scene.png"
    image_pil = Image.open(image_path).convert("RGB")

    # 2. Detect goals from the image
    goal_list = detect_goals_from_image(image_pil)
    print("Goals list for planning:", goal_list)

    # 3. Define the global model IDs(task) for the multi-agent inference
    # global_model_ids = [['EnvConveyor2DNS-RobotPlanarDisk']]
    global_model_ids = [['EnvDropRegion2DNS-RobotPlanarDisk']]
    # global_model_ids = [['EnvRoomMap2DNS-RobotPlanarDisk']]
    # global_model_ids = [['EnvShelf2DNS-RobotPlanarDisk']]

    # 4. Run the multi-agent inference to get trajectories
    paths_l = run_inference(goal_list = goal_list, global_model_ids = global_model_ids)

    # 5. Visualize the generated trajectories
    output_path="tasks/drop_region/results/output_trajectory.png"
    plot_trajectories_on_image(trajectories=paths_l, 
                               background_img=image_pil, 
                               output_path=output_path,
                               line_width=3,
                               state_dot_size=3,
                               start_dot_size=12,
                               end_dot_size=12)
    
    print(f"Final trajectory visualization saved to: {output_path}")