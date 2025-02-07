import matplotlib.pyplot as plt

def visualize_distances(distances):
    current_idx = [0]  # Using a list to allow modifications in the inner function

    fig, ax = plt.subplots()
    img = ax.imshow(distances[current_idx[0]])
    plt.colorbar(img)

    def on_key(event):
        if event.key == 'right':
            current_idx[0] = (current_idx[0] + 1) % len(distances)
        elif event.key == 'left':
            current_idx[0] = (current_idx[0] - 1) % len(distances)
        else:
            return
        # Update image data and redraw
        img.set_data(distances[current_idx[0]])
        ax.set_title(f"Map {current_idx[0]+1}/{len(distances)}")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)
    ax.set_title(f"Map {current_idx[0]+1}/{len(distances)}")
    plt.show()