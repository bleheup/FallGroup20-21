from gym.envs.classic_control import rendering
import json

# TODO: Implement directly into the Everglades environment,
#       reimplement without OpenAI's pyglet front-end classes (gym.envs.classic_control),
#       and generalize to different maps.
class Renderer(object):
    """
    Allows for rendering of Everglades for both frame collection and viewing.
    Use is similar to a gym environment's render() function, where
    render(observation) is used to render the next state of the env.
    However, this is a class to allow for the map to be initialized before
    rendering begins.

    RENDERING INFO:
        BLUE - Player 1 Controlled
        RED - Player 2 Controlled
        NODE TYPES:
            Dark Gray Nodes - Fortresses
            Gray Nodes - Watchtowers
            Light Gray Nodes - Other
        GROUP TYPES:
            Circle - Controllers
            Triangle - Strikers
            Square - Tanks
        
        Transparent groups are in transit.
        Darkened groups are dead.

    Parameters
    ----------
        map_file - The filepath for the map json to be loaded
        frame_collection - Whether to return RGB arrays each time render() is called (False by default)
        hidden - Whether or not to show the rendering (False by default)
		
	Returns
	-------
		self.viewer.render() - None or RGB array
    """
    def __init__(self, map_file, frame_collection = False, hidden = False):
        # The map
        self.load_map(map_file)
        # The game window
        self.viewer = None
        # Map Params
        self.screen_width = 600
        self.screen_height = 400
        self.node_scaling = 40
        self.node_separation = 3
        self.group_scaling = self.node_scaling / 10
        # [0,1] -> lower values mean increased transparency/darkness
        self.transparency = 0.3
        self.death_darkness = 0.3
        # Group information
        self.num_groups = 12
        # Whether or not to return rgb arrays
        self.mode = 'human' if frame_collection == False else 'rgb_array'
        # Whether or not to display window
        self.hidden = hidden

    def load_map(self,map_file):
        """Loads the map_file.json and gets basic info about it"""
        with open(map_file) as fid:
            self.map_dat = json.load(fid)
        self.num_nodes = len(self.map_dat['nodes'])
        return

    # TODO: Generalize to different maps
    def player2_mirror(self, node_pos):
        """
        Returns the second player's equivalent node id from the first player's perspective.
        This is necessary as the current environment's perspective is equivalent to
        the first player's.
        """
        mirror = [11, 8, 9, 10, 5, 6, 7, 2, 3, 4, 1] # DemoMap layout
        return mirror[node_pos-1]

    def _initialize(self,state):
        """Creates the window for display and initializes the render of Everglades."""
        # Creates the pyglet window
        self.viewer = rendering.Viewer(self.screen_width, self.screen_height)
        # Hides window if necessary
        self.viewer.window.set_visible(not self.hidden)

        # Gets locations for nodes
        self.node_locations = []
        for n in range(self.num_nodes):
            locx = self.map_dat['nodes'][n]['X'] * \
                self.node_scaling*self.node_separation + (self.screen_width/2)
            locy = self.map_dat['nodes'][n]['Y'] * \
                self.node_scaling*self.node_separation + (self.screen_height/2)
            self.node_locations.append((locx, locy))

        # Draws edges between nodes (transit routes)
        for n in range(self.num_nodes):
            node_coords = self.node_locations[n]
            for c in [i['ConnectedID'] for i in self.map_dat['nodes'][n]['Connections']]:
                connect_coords = self.node_locations[c-1]
                line = rendering.Line(node_coords, connect_coords)
                self.viewer.add_geom(line)
        
        # Draws the nodes
        self.nodes = []
        for n in range(self.num_nodes):
            # Creates node filling (done once; indicates type of base)
            rad = self.map_dat['nodes'][n]['Radius']
            fill_node = rendering.make_circle(
                radius=rad*self.node_scaling, filled=True)
            
            if state[0][n*4 + 1] == 1: # colors fortesses
                fill_node.set_color(0.2, 0.2, 0.2)
            elif state[0][n*4 + 2] == 1: # colors watchtowers
                fill_node.set_color(0.5, 0.5, 0.5)
            else: # colors other types of nodes
                fill_node.set_color(0.9, 0.9, 0.9)
            
            # Moves nodes to correct location on map
            node_trans = rendering.Transform(
                translation=self.node_locations[n])
            fill_node.add_attr(node_trans)
            # Adds nodes to the rendering
            self.viewer.add_geom(fill_node)

            # Creates the node borders (will indicate controlling team)
            node = rendering.make_circle(
                radius=rad*self.node_scaling, filled=False)
            # Intializes to show no nodes are controlled
            node.set_color(0, 0, 0)
            node.set_linewidth(self.node_scaling/10)
            node.add_attr(node_trans)
            self.viewer.add_geom(node)
            # Allows for the borders to be updated throughout the game
            self.nodes.append(node)
        
        # Creates player 1 groups (blue)
        self.player1_groups = []
        self.player1_trans = []
        for g in range(self.num_groups):
            # Controller (circle)
            if state[0][g*5 + 46] == 0:
                group = rendering.make_circle(
                    radius=self.group_scaling, filled=True)
            # Striker (triangle)
            elif state[0][g*5 + 46] == 1:
                group = rendering.FilledPolygon([(0, self.group_scaling),
                    (-self.group_scaling, -self.group_scaling),
                    (self.group_scaling, -self.group_scaling)])
            # Tank (square)
            else:
                group = rendering.FilledPolygon([(-self.group_scaling, self.group_scaling),
                    (self.group_scaling, self.group_scaling),
                    (self.group_scaling, -self.group_scaling),
                    (-self.group_scaling, -self.group_scaling)])
            group.set_color(0, 0, 1)
            # Gets group location
            group_node_loc = self.node_locations[int(state[0][g*5 + 45]-1)]
            group_locx = ((g % 6)-2.5) * self.group_scaling * \
                3 + group_node_loc[0]
            group_locy = int(g/6) * self.group_scaling * 3 + \
                group_node_loc[1] + self.group_scaling
            group_trans = rendering.Transform(
                translation=(int(group_locx), int(group_locy)))
            group.add_attr(group_trans)
            # Adds to the window and saves info to allow updating
            self.viewer.add_geom(group)
            self.player1_groups.append(group)
            self.player1_trans.append(group_trans)
        
        # Creates player 2 groups (red)
        self.player2_groups = []
        self.player2_trans = []
        for g in range(self.num_groups):
            # Controller (circle)
            if state[1][g*5 + 46] == 0:
                group = rendering.make_circle(
                    radius=self.group_scaling, filled=True)
            # Striker (triangle)
            elif state[1][g*5 + 46] == 1:
                group = rendering.FilledPolygon([(0, self.group_scaling),
                    (-self.group_scaling, -self.group_scaling),
                    (self.group_scaling, -self.group_scaling)])
            # Tank (square)
            else:
                group = rendering.FilledPolygon([(-self.group_scaling, self.group_scaling),
                    (self.group_scaling, self.group_scaling),
                    (self.group_scaling, -self.group_scaling),
                    (-self.group_scaling, -self.group_scaling)])
            group.set_color(1, 0, 0)
            # Gets group location
            group_node_loc = self.node_locations[self.player2_mirror(
                int(state[1][g*5 + 45]))-1]
            group_locx = ((g % 6)-2.5) * self.group_scaling * \
                3 + group_node_loc[0]
            group_locy = int(g/6) * -self.group_scaling * 3 + \
                group_node_loc[1] - self.group_scaling
            group_trans = rendering.Transform(
                translation=(int(group_locx), int(group_locy)))
            group.add_attr(group_trans)
            # Adds to the window and saves info to allow updating
            self.viewer.add_geom(group)
            self.player2_groups.append(group)
            self.player2_trans.append(group_trans)


    def render(self, state):
        """Renders the state given for the Everglades map"""
        # Initialializes Screen and Nodes
        if self.viewer is None:
            self._initialize(state)

        # Shows what team controls which node
        for n in range(self.num_nodes):
            # self owned
            percent_controlled = state[0][n*4+3]
            if percent_controlled > 0:
                self.nodes[n].set_color(0, 0, 1)
            # enemy owned
            elif percent_controlled < 0:
                self.nodes[n].set_color(1, 0, 0)
            # neutral
            else:
                self.nodes[n].set_color(0, 0, 0)

        # Changes player 1 groups
        for g in range(self.num_groups):
            # darkens if group is dead
            if int(state[0][g*5 + 49]) == 0:
                self.player1_groups[g]._color.vec4 = (
                    0, 0, self.death_darkness, 1.0)
            # redraws player 1 group
            else:
                # Changes location
                group_node_loc = self.node_locations[int(state[0][g*5 + 45]-1)]
                group_locx = ((g % 6)-2.5) * self.group_scaling * \
                    3 + group_node_loc[0]
                group_locy = int(g/6) * self.group_scaling * 3 + \
                    group_node_loc[1] + self.group_scaling
                self.player1_trans[g].set_translation(group_locx, group_locy)
                # increases transparency if in transit
                if int(state[0][g*5 + 48]) == 1:
                    self.player1_groups[g]._color.vec4 = (
                        0, 0, 1, self.transparency)
                # reverts to opaque
                else:
                    self.player1_groups[g]._color.vec4 = (0, 0, 1, 1.0)

        # Changes player 2 groups
        for g in range(self.num_groups):
            # darkens if group is dead
            if int(state[1][g*5 + 49]) == 0:
                self.player2_groups[g]._color.vec4 = (
                    self.death_darkness, 0, 0, 1.0)
            # redraws player 2 group
            else:
                # Changes location
                group_node_loc = self.node_locations[self.player2_mirror(
                    int(state[1][g*5 + 45]))-1]
                group_locx = ((g % 6)-2.5) * self.group_scaling * \
                    3 + group_node_loc[0]
                group_locy = int(g/6) * -self.group_scaling * 3 + \
                    group_node_loc[1] - self.group_scaling
                self.player2_trans[g].set_translation(group_locx, group_locy)
                # increases transparency if in transit
                if int(state[1][g*5 + 48]) == 1:
                    self.player2_groups[g]._color.vec4 = (
                        1, 0, 0, self.transparency)
                # reverts to opaque
                else:
                    self.player2_groups[g]._color.vec4 = (1, 0, 0, 1.0)

        # Performs render
        return self.viewer.render(return_rgb_array=self.mode == 'rgb_array')