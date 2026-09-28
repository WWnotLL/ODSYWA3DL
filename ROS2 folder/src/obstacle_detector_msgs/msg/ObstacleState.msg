uint8 CLEAR = 0
uint8 UNCHECKED = 1
uint8 OBSTACLE = 2

std_msgs/Header header
uint8 state
bool stop
bool obstacle_found
float32 nearest_distance_m
float32 checked_range_m
Obstacle[] obstacles
