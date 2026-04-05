import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import json
import math
import cv2
import io
import os

class StyledPlanGenerator:
    def __init__(self, width=900, height=900, output_dir="Dataset"):
        self.width = width
        self.height = height
        self.dpi = 100
        
        # --- 1. GLOBAL STYLE SETTINGS (Per Plan Consistency) ---
        # Wall Style
        self.wall_thickness = random.choice([2, 3, 4, 5])
        self.wall_color = random.choice(['#000000', '#2F2F2F', '#000050']) # Black, Dark Gray, Navy
        self.fill_style = random.choice(['solid_black', 'solid_grey', 'hatch', 'empty'])
        
        # Dimension Style
        self.arrow_style = random.choice(['<->', '|-|', '-|>', '<|-|>', '->'])
        self.dim_color = '#000000' # Dims usually stay black even if walls vary
        # Distance from wall: Tight (25px) vs Loose (60px)
        self.base_offset = random.randint(25, 65) 
        
        # Font Style
        self.font_family = random.choice(['sans-serif', 'serif', 'monospace'])
        self.font_size = random.randint(8, 11)
        
        # Unit System (The "7.5", "mm", "inches" request)
        # Options: 'mm' (4500), 'm' (4.5), 'inch' (120"), 'ft' (10'6")
        self.unit_mode = random.choice(['mm', 'mm_suffix', 'm', 'inch', 'ft'])
        
        # Plan Complexity (1-2 rooms vs many)
        # 1 = Studio (1-2 rooms), 2 = Simple (3-4 rooms), 4 = Complex (8+ rooms)
        self.complexity_depth = random.choice([1, 2, 4, 4, 4]) 

        # Setup
        self.img_dir = os.path.join(output_dir, "images")
        self.json_dir = os.path.join(output_dir, "jsons")
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)
        
        # Data
        self.walls = []      
        self.windows = [] 
        self.dimensions = [] 
        self.rooms = [] 
        self.text_bboxes = [] 
        
        self.wall_id_counter = 100
        self.window_id_counter = 2000
        self.dim_id_counter = 1
        
        self.fig = plt.figure(figsize=(width/self.dpi, height/self.dpi), dpi=self.dpi)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.axis('off')

    def _format_value(self, px_length):
        """Converts pixel length to the plan's specific unit string."""
        # Baseline: 10px = 100mm
        mm_val = int(px_length * 10)
        
        if self.unit_mode == 'mm':
            return str(mm_val)
        elif self.unit_mode == 'mm_suffix':
            return f"{mm_val} mm"
        elif self.unit_mode == 'm':
            return f"{round(mm_val / 1000, 1)}m" # 4.5m
        elif self.unit_mode == 'inch':
            return f"{int(mm_val / 25.4)}\"" # 120"
        elif self.unit_mode == 'ft':
            total_inches = int(mm_val / 25.4)
            ft = total_inches // 12
            inch = total_inches % 12
            return f"{ft}'{inch}\""
        return str(mm_val)

    def generate_layout(self):
        margin = 150
        initial_rect = (margin, margin, self.width - 2*margin, self.height - 2*margin)
        
        # --- 5. USE VARIABLE DEPTH ---
        raw_rooms = self._recursive_split(initial_rect, depth=self.complexity_depth)
        
        # Only cull shapes if we have enough rooms (don't delete the only room!)
        if len(raw_rooms) > 3:
            center_x, center_y = self.width/2, self.height/2
            raw_rooms.sort(key=lambda r: ((r[0]-center_x)**2 + (r[1]-center_y)**2), reverse=True)
            num_to_remove = random.randint(1, max(1, int(len(raw_rooms) * 0.4)))
            self.rooms = raw_rooms[num_to_remove:]
        else:
            self.rooms = raw_rooms

        room_labels = ["LIVING", "KITCHEN", "DINING", "BEDROOM", "MASTER BED", "BATH", "STORE", "STUDY"]
        random.shuffle(room_labels)
        for i, room in enumerate(self.rooms):
            label = room_labels[i % len(room_labels)]
            self.rooms[i] = (*room, label)

        for (x, y, w, h, _) in self.rooms:
            room_cx, room_cy = x + w/2, y + h/2
            self._add_wall(x, y, x + w, y, room_cx, room_cy)         
            self._add_wall(x + w, y, x + w, y+h, room_cx, room_cy) 
            self._add_wall(x + w, y + h, x, y+h, room_cx, room_cy) 
            self._add_wall(x, y + h, x, y, room_cx, room_cy)       

    def _recursive_split(self, rect, depth):
        x, y, w, h = rect
        # Stop if depth reached OR if room is too small
        if depth == 0 or w < 160 or h < 160: return [rect]
        
        if w > h:
            split = random.randint(int(w * 0.4), int(w * 0.6))
            return self._recursive_split((x, y, split, h), depth-1) + self._recursive_split((x+split, y, w-split, h), depth-1)
        else:
            split = random.randint(int(h * 0.4), int(h * 0.6))
            return self._recursive_split((x, y, w, split), depth-1) + self._recursive_split((x, y+split, w, h-split), depth-1)

    def _add_wall(self, x1, y1, x2, y2, rcx, rcy):
        if x1 > x2 or y1 > y2: x1, x2, y1, y2 = x2, x1, y2, y1
        for w in self.walls:
            wx1, wy1, wx2, wy2 = w['coords']
            if abs(wx1-x1)<5 and abs(wx2-x2)<5 and abs(wy1-y1)<5 and abs(wy2-y2)<5:
                w['is_shared'] = True 
                return 
        self.walls.append({
            "id": self.wall_id_counter, "coords": [int(x1), int(y1), int(x2), int(y2)], 
            "is_shared": False, "has_opening": False,
            "room_center": (rcx, rcy)
        })
        self.wall_id_counter += 1

    def draw_structure(self):
        # --- 1. USE GLOBAL WALL STYLES ---
        for w in self.walls:
            x1, y1, x2, y2 = w['coords']
            is_horiz = abs(y1 - y2) < 5
            
            # Wall Outline
            self.ax.plot([x1, x2], [y1, y2], color=self.wall_color, lw=self.wall_thickness, zorder=10)
            
            thick = self.wall_thickness * 2.5 # Fill proportional to thickness
            if self.fill_style != 'empty':
                if is_horiz: rect = patches.Rectangle((x1, y1-thick/2), x2-x1, thick, angle=0)
                else: rect = patches.Rectangle((x1-thick/2, y1), thick, y2-y1, angle=0)
                
                if self.fill_style == 'solid_black': rect.set_facecolor(self.wall_color)
                elif self.fill_style == 'solid_grey': rect.set_facecolor('#808080')
                elif self.fill_style == 'hatch':
                    rect.set_facecolor('white'); rect.set_hatch('///'); rect.set_edgecolor(self.wall_color); rect.set_linewidth(0)
                rect.set_zorder(1)
                self.ax.add_patch(rect)

    def add_grid_lines(self):
        x_starts = sorted(list(set([w['coords'][0] for w in self.walls])))[::2][:3]
        y_starts = sorted(list(set([w['coords'][1] for w in self.walls])))[::2][:3]
        labels = ['A', 'B', 'C', '1', '2', '3']
        for i, x in enumerate(x_starts):
            self.ax.plot([x, x], [50, self.height-50], color='#A0A0A0', ls='-.', lw=0.5, zorder=0)
            self.ax.text(x, self.height-30, labels[i%3], ha='center', va='center', fontsize=8, bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec="black", lw=0.5))
        for i, y in enumerate(y_starts):
            self.ax.plot([50, self.width-50], [y, y], color='#A0A0A0', ls='-.', lw=0.5, zorder=0)
            self.ax.text(30, y, labels[i%3+3], ha='center', va='center', fontsize=8, bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec="black", lw=0.5))

    def add_room_labels_and_furniture(self):
        for room in self.rooms:
            x, y, w, h, label = room
            cx, cy = x + w/2, y + h/2
            # --- 3. USE GLOBAL FONT ---
            self.ax.text(cx, cy, label, ha='center', va='center', fontsize=self.font_size, 
                         family=self.font_family, fontweight='bold', color='#404040', zorder=5)
            
            # Format area using the global unit system
            area_w = self._format_value(w)
            area_h = self._format_value(h)
            self.ax.text(cx, cy-15, f"{area_w} x {area_h}", ha='center', va='center', 
                         fontsize=self.font_size-2, family=self.font_family, color='#404040', zorder=5)

            if min(w, h) < 60: continue 
            # (Furniture drawing code same as before, omitted for brevity)
            # You can copy the furniture block from the previous script here if needed

    def add_windows(self):
        # (Standard window logic)
        standard_sizes = [30, 45, 60, 75, 90] 
        for w in self.walls:
            if w['is_shared'] or w['has_opening']: continue 
            if random.random() > 0.4: 
                x1, y1, x2, y2 = w['coords']
                is_horiz = abs(y1 - y2) < 5
                wall_len = abs(x2-x1) if is_horiz else abs(y2-y1)
                valid_sizes = [s for s in standard_sizes if s < (wall_len - 20)]
                if not valid_sizes: continue 
                win_len = random.choice(valid_sizes)
                mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
                if is_horiz:
                    self.ax.add_patch(patches.Rectangle((mid_x-win_len/2, mid_y-3), win_len, 6, lw=1, ec='#999', fc='white', zorder=11))
                    win_coords = [mid_x-win_len/2, mid_y, mid_x+win_len/2, mid_y]
                else:
                    self.ax.add_patch(patches.Rectangle((mid_x-3, mid_y-win_len/2), 6, win_len, lw=1, ec='#999', fc='white', zorder=11))
                    win_coords = [mid_x, mid_y-win_len/2, mid_x, mid_y+win_len/2]
                w['has_opening'] = True
                self.windows.append({"id": self.window_id_counter, "wall_id": w['id'], "coords": [int(c) for c in win_coords], "length": int(win_len)})
                self.window_id_counter += 1

    def add_doors(self):
        # (Standard door logic, using self.wall_thickness)
        door_size = 35
        for room in self.rooms:
            if random.random() > 0.6: continue 
            x, y, w_room, h_room, _ = room
            candidates = [([x, x+w_room], [y+h_room, y+h_room], 0), ([x+w_room, x+w_room], [y, y+h_room], 1), ([x, x+w_room], [y, y], 2), ([x, x], [y, y+h_room], 3)]
            random.shuffle(candidates)
            target_wall, wall_idx = None, -1
            for (wx, wy, widx) in candidates:
                cx, cy = (wx[0]+wx[1])/2, (wy[0]+wy[1])/2
                for existing_wall in self.walls:
                    ex1, ey1, ex2, ey2 = existing_wall['coords']
                    if (min(ex1,ex2) <= cx <= max(ex1,ex2)) and (min(ey1,ey2) <= cy <= max(ey1,ey2)):
                        if not existing_wall['has_opening']: target_wall, wall_idx = existing_wall, widx; break
                if target_wall: break
            if not target_wall: continue
            wx1, wy1, wx2, wy2 = target_wall['coords']
            # Draw door using self.wall_color
            if wall_idx in [0, 2]: 
                 dx = random.randint(int(wx1)+5, int(wx2)-door_size-5); dy = wy1
                 self.ax.plot([dx, dx+door_size], [dy, dy], color='white', lw=self.wall_thickness+1, zorder=11) # Cut
                 swing_start = (dx, dy) if wall_idx == 2 else (dx+door_size, dy)
                 theta1, theta2 = (0, 90) if wall_idx == 2 else (180, 270)
                 self.ax.add_patch(patches.Arc(swing_start, door_size*2, door_size*2, theta1=theta1, theta2=theta2, color=self.wall_color, lw=1, zorder=11))
                 self.ax.plot([swing_start[0], swing_start[0] + (door_size * math.cos(math.radians(theta2)))], [swing_start[1], swing_start[1] + (door_size * math.sin(math.radians(theta2)))], color=self.wall_color, lw=1, zorder=11)
            else: 
                 dx = wx1; dy = random.randint(int(wy1)+5, int(wy2)-door_size-5)
                 self.ax.plot([dx, dx], [dy, dy+door_size], color='white', lw=self.wall_thickness+1, zorder=11)
                 swing_start = (dx, dy) if wall_idx == 1 else (dx, dy+door_size)
                 theta1, theta2 = (90, 180) if wall_idx == 1 else (270, 360)
                 self.ax.add_patch(patches.Arc(swing_start, door_size*2, door_size*2, theta1=theta1, theta2=theta2, color=self.wall_color, lw=1, zorder=11))
                 self.ax.plot([swing_start[0], swing_start[0] + (door_size * math.cos(math.radians(theta2)))], [swing_start[1], swing_start[1] + (door_size * math.sin(math.radians(theta2)))], color=self.wall_color, lw=1, zorder=11)
            target_wall['has_opening'] = True

    def _line_crosses_walls(self, p1, p2, my_wall_id):
        x1, y1 = p1; x2, y2 = p2
        for w in self.walls:
            if w['id'] == my_wall_id: continue 
            wx1, wy1, wx2, wy2 = w['coords']
            if min(x1,x2) > max(wx1,wx2) or max(x1,x2) < min(wx1,wx2): continue
            if min(y1,y2) > max(wy1,wy2) or max(y1,y2) < min(wy1,wy2): continue
            is_dim_horiz = abs(y1-y2) < 1; is_wall_horiz = abs(wy1-wy2) < 5
            if is_dim_horiz and not is_wall_horiz:
                if min(x1,x2) < wx1 < max(x1,x2) and min(wy1,wy2) < y1 < max(wy1,wy2): return True
            elif not is_dim_horiz and is_wall_horiz:
                if min(y1,y2) < wy1 < max(y1,y2) and min(wx1,wx2) < x1 < max(wx1,wx2): return True
        return False

    def add_smart_dimensions(self):
        strategy = random.choice(['detailed', 'simple', 'mixed'])
        for w in self.walls:
            if w['is_shared']: continue 
            if random.random() > 0.7: continue
            
            wall_strat = strategy
            if strategy == 'mixed': wall_strat = 'detailed' if random.random() > 0.5 else 'simple'
            
            attached_window = None
            for win in self.windows:
                if win['wall_id'] == w['id']: attached_window = win; break
            
            wx1, wy1, wx2, wy2 = w["coords"]
            is_horizontal = abs(wy1 - wy2) < 5
            
            # --- 2. USE GLOBAL OFFSET ---
            # Slight jitter (+- 5) around the plan's base offset
            offset_dist = self.base_offset + random.randint(-5, 5)
            
            rcx, rcy = w['room_center']
            wmx, wmy = (wx1+wx2)/2, (wy1+wy2)/2
            
            if is_horizontal:
                direction = 1 if wmy > rcy else -1
                base_dx, base_dy = 0, offset_dist * direction
            else:
                direction = 1 if wmx > rcx else -1
                base_dx, base_dy = offset_dist * direction, 0

            # Chain Logic
            if attached_window and wall_strat == 'detailed':
                win_x1, win_y1, win_x2, win_y2 = attached_window['coords']
                if is_horizontal:
                    points = sorted([wx1, wx2, win_x1, win_x2])
                    segments = [(points[0], wy1, points[1], wy1, "wall_id", w['id']), (points[1], wy1, points[2], wy1, "window_id", attached_window['id']), (points[2], wy1, points[3], wy1, "wall_id", w['id'])]
                else:
                    points = sorted([wy1, wy2, win_y1, win_y2])
                    segments = [(wx1, points[0], wx1, points[1], "wall_id", w['id']), (wx1, points[1], wx1, points[2], "window_id", attached_window['id']), (wx1, points[2], wx1, points[3], "wall_id", w['id'])]
                
                for (sx1, sy1, sx2, sy2, link_type, link_id) in segments:
                    length = math.hypot(sx2-sx1, sy2-sy1)
                    if length < 5: continue
                    # --- 4. USE FORMATTED VALUE ---
                    val_str = self._format_value(length)
                    self._draw_dimension_safe(sx1+base_dx, sy1+base_dy, sx2+base_dx, sy2+base_dy, val_str, link_type, link_id, w['id'])
            else:
                length = math.hypot(wx2-wx1, wy2-wy1)
                val_str = self._format_value(length)
                self._draw_dimension_safe(wx1+base_dx, wy1+base_dy, wx2+base_dx, wy2+base_dy, val_str, "wall_id", w['id'], w['id'])

    def _check_collision(self, new_bbox):
        nx1, ny1, nx2, ny2 = new_bbox
        nx1, ny1, nx2, ny2 = nx1-2, ny1-2, nx2+2, ny2+2
        for (ex1, ey1, ex2, ey2) in self.text_bboxes:
            if (nx1 < ex2 and nx2 > ex1 and ny1 < ey2 and ny2 > ey1): return True
        return False

    def _draw_dimension_safe(self, x1, y1, x2, y2, text_val, link_type, link_id, owner_wall_id):
        if self._line_crosses_walls((x1, y1), (x2, y2), owner_wall_id): return 

        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        is_vert = abs(x1 - x2) < 1
        rotation = 90 if is_vert else 0
        char_w, char_h = 8, 10
        text_len_px = len(text_val) * char_w
        if is_vert: sim_bbox = [mid_x-char_h/2, mid_y-text_len_px/2, mid_x+char_h/2, mid_y+text_len_px/2]
        else: sim_bbox = [mid_x-text_len_px/2, mid_y-char_h/2, mid_x+text_len_px/2, mid_y+char_h/2]
        if self._check_collision(sim_bbox): return 

        # --- 2. USE GLOBAL ARROW STYLE ---
        self.ax.annotate("", xy=(x1, y1), xytext=(x2, y2), arrowprops=dict(arrowstyle=self.arrow_style, lw=0.7, color=self.dim_color), zorder=12)
        
        pos_mode = random.choices(['inline', 'above', 'below'], weights=[0.4, 0.3, 0.3])[0]
        shift = 9
        bg_alpha = 0.0
        if pos_mode == 'inline': tx, ty = mid_x, mid_y; bg_alpha = 1.0 
        elif pos_mode == 'above': tx = mid_x - shift if is_vert else mid_x; ty = mid_y if is_vert else mid_y + shift
        else: tx = mid_x + shift if is_vert else mid_x; ty = mid_y if is_vert else mid_y - shift

        # --- 3. USE GLOBAL FONT ---
        t = self.ax.text(tx, ty, text_val, rotation=rotation, ha='center', va='center',
            fontsize=self.font_size, family=self.font_family, color=self.dim_color,
            bbox=dict(facecolor='white', edgecolor='none', alpha=bg_alpha, pad=1), zorder=13)

        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        bbox = t.get_window_extent(renderer)
        self.text_bboxes.append([bbox.x0, bbox.y0, bbox.x1, bbox.y1])
        
        dim_entry = {
            "id": self.dim_id_counter, "val": text_val, "position": pos_mode,
            "bbox": [int(bbox.x0), int(self.height - bbox.y1), int(bbox.x1), int(self.height - bbox.y0)],
            "dim_line": [int(x1), int(y1), int(x2), int(y2)]
        }
        dim_entry[link_type] = link_id
        self.dimensions.append(dim_entry)
        self.dim_id_counter += 1

    def generate(self, filename):
        self.generate_layout()
        self.add_grid_lines()
        self.draw_structure()
        self.add_room_labels_and_furniture()
        self.add_windows()   
        self.add_doors()      
        self.add_smart_dimensions()
        
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), 1)
        
        img_path = os.path.join(self.img_dir, f"{filename}.jpg")
        cv2.imwrite(img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        output_data = {"image_file": f"{filename}.jpg", "walls": self.walls, "windows": self.windows, "dimensions": self.dimensions}
        with open(os.path.join(self.json_dir, f"{filename}.json"), "w") as f:
            json.dump(output_data, f, indent=2)
        plt.close(self.fig)
        return f"Created {filename}"

if __name__ == "__main__":
    for i in range(10):
        gen = StyledPlanGenerator(width=900, height=900)
        print(gen.generate(f"synthetic_plan_{i}"))