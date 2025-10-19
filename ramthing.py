# pip install gym-retro gymnasium pygame
import numpy as np
import pygame
import retro
import re

# ---- Key bindings you want on keyboard
KEYS = {
    "B": pygame.K_z,       # jump
    "Y": pygame.K_x,       # run/roll
    "Select": pygame.K_RSHIFT,
    "Start": pygame.K_RETURN,
    "Up": pygame.K_UP,
    "Down": pygame.K_DOWN,
    "Left": pygame.K_LEFT,
    "Right": pygame.K_RIGHT,
    "A": pygame.K_c,
    "X": pygame.K_s,
    "L": pygame.K_a,
    "R": pygame.K_d,
}

# Fuzzy names to catch variations (e.g., 'RIGHT', 'D-Pad Right', etc.)
ALIASES = {
    "B":     ["B"],
    "Y":     ["Y"],
    "Select":["SELECT"],
    "Start": ["START"],
    "Up":    ["UP"],
    "Down":  ["DOWN"],
    "Left":  ["LEFT"],
    "Right": ["RIGHT"],
    "A":     ["A"],
    "X":     ["X"],
    "L":     ["L", "L1"],
    "R":     ["R", "R1"],
}

def normalize(s: str) -> str:
    return re.sub(r"\s+|-|_", "", s).upper()

def build_button_index_map(env_buttons):
    """
    Returns: idx_map: dict[canonical_name] = index_in_env_buttons
    and prints the env buttons for debugging.
    """
    print("env.buttons =", env_buttons)
    # Normalize env button names
    norm_env = [normalize(b) for b in env_buttons]
    idx_map = {}

    for canon, candidates in ALIASES.items():
        found = None
        for i, nb in enumerate(norm_env):
            for cand in candidates:
                if normalize(cand) == nb:
                    found = i
                    break
            if found is not None:
                break
        # If not exact: try contains (handles 'DPADRIGHT', 'DPAD_RIGHT', etc.)
        if found is None:
            for i, nb in enumerate(norm_env):
                if any(normalize(cand) in nb for cand in candidates):
                    found = i
                    break
        if found is not None:
            idx_map[canon] = found
    return idx_map

def main():
    env = retro.make("DonkeyKongCountry-Snes", use_restricted_actions=retro.Actions.ALL)
    buttons = getattr(env, "buttons", ['B','Y','Select','Start','Up','Down','Left','Right','A','X','L','R'])
    idx_map = build_button_index_map(buttons)

    # Minimal sanity: require that at least Right/Left exist
    if "Right" not in idx_map or "Left" not in idx_map:
        print("Could not identify Right/Left in env.buttons. Indices found:", idx_map)
        print("Paste the printed env.buttons here and I’ll map it for you.")
        return

    obs, info = env.reset()
    prev_ram = env.get_ram().copy()

    pygame.init()
    pygame.display.set_caption("DKC – manual control (ESC quits)")
    screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
    clock = pygame.time.Clock()

    running = True
    while running:
        # Pump events so pygame updates key states
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            break

        # Build action vector each frame
        action = np.zeros(len(buttons), dtype=np.int8)
        for canon, key in KEYS.items():
            if canon in idx_map and keys[key]:
                action[idx_map[canon]] = 1

        # Debug: show when arrows are held & what index they hit
        if keys[pygame.K_RIGHT]:
            print(f"Holding RIGHT -> index {idx_map['Right']}")
        if keys[pygame.K_LEFT]:
            print(f"Holding LEFT  -> index {idx_map['Left']}")
        if keys[pygame.K_UP]:
            print(f"Holding UP    -> index {idx_map['Up']}")
        if keys[pygame.K_DOWN]:
            print(f"Holding DOWN  -> index {idx_map['Down']}")

        # Step the env
        obs, reward, terminated, truncated, info = env.step(action)

        # RAM diff (first 20 indices)
        ram = env.get_ram()
        diff_idxs = np.where(ram != prev_ram)[0]
        if diff_idxs.size:
            print("ΔRAM idx:", diff_idxs[:20])
        prev_ram = ram.copy()

        # Render (new API)
        env.render()              # updates internal screen
        frame = env.get_screen()  # fetch numpy RGB frame

        # Blit to pygame
        if frame is not None:
            h, w, _ = frame.shape
            win_w, win_h = screen.get_size()
            scale = min(win_w / w, win_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            surf = pygame.transform.smoothscale(surf, (new_w, new_h))
            screen.fill((0, 0, 0))
            screen.blit(surf, ((win_w-new_w)//2, (win_h-new_h)//2))
            pygame.display.flip()

        if terminated or truncated:
            obs, info = env.reset()
            prev_ram = env.get_ram().copy()

        clock.tick(60)

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()
#firstbasic enviornment

#Observation Space: What can your agent see? (images, numbers, structured data, etc.)
