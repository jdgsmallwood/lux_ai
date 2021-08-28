from kaggle_environments import make

# Important imports
from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES, Position
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux.game_objects import Player, City, Unit, CityTile
from lux import annotate
from queue import PriorityQueue
from collections import defaultdict
import random
import math
import sys
from abc import ABC, abstractmethod

DIRECTIONS = Constants.DIRECTIONS

#env = make("lux_ai_2021", configuration={"seed": 562124210, "loglevel": 2, "annotations": True}, debug=True)

# this snippet finds all resources stored on the map and puts them into a list so we can search over them
def find_resources(game_state, researched_coal=False, researched_uranium=False):
    resource_tiles = []
    width, height = game_state.map_width, game_state.map_height
    for y in range(height):
        for x in range(width):
            cell = game_state.map.get_cell(x, y)
            if cell.has_resource():
                if cell.resource.type == Constants.RESOURCE_TYPES.COAL and not researched_coal: continue
                if cell.resource.type == Constants.RESOURCE_TYPES.URANIUM and not researched_uranium: continue
                resource_tiles.append(cell)
    return resource_tiles


def find_resources_in_range(game_state, unit, unit_range, researched_coal=False, researched_uranium=False):
    resource_tiles = find_resources(game_state, researched_coal, researched_uranium)
    resource_tiles_in_range = [(tile, unit.pos.distance_to(tile.pos)) for tile in resource_tiles if unit.pos.distance_to(tile.pos) < unit_range]
    return resource_tiles_in_range


# the next snippet finds the closest resources that we can mine given position on a map
def find_closest_resources(pos, player, game_state):
    resource_tiles = find_resources(game_state)
    
    closest_dist = math.inf
    closest_resource_tile = None
    for resource_tile in resource_tiles:
        # we skip over resources that we can't mine due to not having researched them
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.COAL and not player.researched_coal(): continue
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.URANIUM and not player.researched_uranium(): continue
        dist = resource_tile.pos.distance_to(pos)
        if dist < closest_dist:
            closest_dist = dist
            closest_resource_tile = resource_tile
    return closest_resource_tile


def find_closest_city_tile(pos, player, city_id=None):
    """
    This will allow to find the closest city tile and optionally also specify which city they want to go to.
    If it is none, then start moving towards the closest tile.
    """
    closest_city_tile = None
    if len(player.cities) > 0:
        closest_dist = math.inf
        # the cities are stored as a dictionary mapping city id to the city object, which has a citytiles field that
        # contains the information of all citytiles in that city
        for k, city in player.cities.items():
            if city_id is None or city_id == k:
                for city_tile in city.citytiles:
                    dist = city_tile.pos.distance_to(pos)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_city_tile = city_tile
        
        if closest_city_tile is None:
            # If you cant' find that particular city (because it doesn't exist anymore) then just get the 
            # closest tile.
            closest_city_tile = find_closest_city_tile(pos, player)
    return closest_city_tile


def get_number_of_city_tiles(player: Player):
    return len(get_all_city_tiles(player))


def get_all_city_tiles(player: Player):
    """
    This will return a list of all city tiles for a particular player.
    """
    output = []
    for id, city in player.cities.items():
        output += city.citytiles
    return output


# This will get the adjacent squares to a particular tile
def get_adjacent_cells(pos: Position, game_state):
    """
    This will just get the cells which are north/south/east/west of the given cell. 
    """
    tile_x = pos.x
    tile_y = pos.y
    
    adjacent_cells = []
    for i, j in zip([-1, 0, 0, 1], [0, -1, 1, 0]):
        adjacent_cells.append(game_state.map.get_cell(tile_x + i, tile_y + j))
    
    return adjacent_cells


# Now I want to get all the cells where I could potentially build my city.
def get_adjacent_cells_to_build(pos: Position, game_state):
    """
    Get all of the cells where I could potentially build my city. In order to build a city there needs to be 
    nothing on the cell.
    """
    adjacent_cells = get_adjacent_cells(pos, game_state)
    return [cell for cell in adjacent_cells if not cell.has_resource() and cell.citytile is None and cell.road == 0]


# This will find a good place to expand the city
def find_best_place_to_expand_city(city_id: int, player: Player, game_state, unit: Unit) -> Cell:
    """
    This will find a good place to expand the city. Initially it will just return a possible place at random,
    however in future iterations I am planning this to minimize the upkeep of the city.
    
    Should change this to be the closest of the possible tiles initially.
    """
    city_tiles = player.cities[city_id].citytiles
    adjacent_tiles = [v for city_tile in city_tiles for v in get_adjacent_cells_to_build(city_tile.pos, game_state) ]
    
    closest_tile = None
    closest_distance = 1000
    
    for tile in adjacent_tiles:
        dis = unit.pos.distance_to(tile.pos)
        if dis < closest_distance:
            closest_distance = dis
            closest_tile = tile
    return closest_tile
    
    
class Node():
    """
    This is a wrapper for A* pathfinding.
    """
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position
    
    def __lt__(self, other): 
        return self.f < other.f

    
def direction_to(start_pos: Position, end_pos: Position, already_used_squares, game_state, avoid_cities=False, is_night=False, citytiles=None) -> DIRECTIONS:
    """
    Return closest position to end_pos. If avoid_cities is True then it will go around cities. 
    
    Collision avoidance - don't go to a square that's already been claimed by someone else.
    """
    closest_dist = start_pos.distance_to(end_pos)
    closest_dir = DIRECTIONS.CENTER
    
    width, height = game_state.map.width, game_state.map.height
    
    avoid_squares = already_used_squares.copy()
    
    if avoid_cities:
        avoid_squares += [tile.pos for tile in citytiles]
    
    if end_pos in avoid_squares or start_pos == end_pos:
        new_cell_obj = game_state.map.get_cell_by_pos(start_pos)
        if new_cell_obj.citytile is None:
            already_used_squares.append(start_pos)
        return DIRECTIONS.CENTER
    
    ### Implementation of A* pathfinding.
    ### f = g + h where g = distance from start & h = estimated distance to end.
    initial_square = Node(parent=None, position=start_pos)
    initial_square.g = 0
    initial_square.h = closest_dist
    initial_square.f = closest_dist
    
    end_square = Node(parent=None, position=end_pos)
    end_square.f = closest_dist
    end_square.g = closest_dist
    end_square.h = 0
    
    # Using priority queues for speed improvements.
    open_list = PriorityQueue()
    open_list.put(initial_square)
    closed_list = []
    
    final_path = None
    
    while not open_list.empty():
        
        # Get the lowest f-value on the current list. This will pop from the openlist.
        # Priority queue will get the lowest priority value next.
        current_node = open_list.get()
        # If it's already in the closesd list - this means that there was a better implementation
        # of this position with lower g value, so just skip this one.
        if current_node.position in closed_list:
            continue
        # Move from open to closed list.
        closed_list.append(current_node.position)
        
        if current_node == end_square:
            path = []
            current = current_node
            while current is not None:
        
                path.append(current.position)
                current = current.parent
            final_path = path[::-1]
            break

        children = []
        for x, y in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            next_pos = Position(current_node.position.x + x, current_node.position.y + y)
            # Check still in map.
            if next_pos.x < 0 | next_pos.y < 0 | next_pos.x >= height | next_pos.y >= width:
                continue
            
            # Check walkable
            if next_pos in avoid_squares:
                continue
            
            next_node = Node(current_node, next_pos)
            children.append(next_node)
        
        for child in children:
            # If it's already in the list then ignore it.
            if child.position in closed_list:
                continue
            
            child.g = current_node.position.distance_to(child.position)
            child.h = child.position.distance_to(end_square.position)
            child.f = child.g + child.h
            
            #open_nodes_same_pos = [node for priority, node in open_list if node.position == child.position]
            #if len(open_nodes_same_pos) > 0 and child.g > open_nodes_same_pos[0].g:
            #    continue
       
            open_list.put(child)

    new_cell = final_path[1]
    new_cell_obj = game_state.map.get_cell_by_pos(new_cell)
    # We can stack units on citytiles, so only add to avoid squares if it's not a citytile.
    if new_cell_obj.citytile is None:
        already_used_squares.append(new_cell)
    closest_dir = start_pos.direction_to(new_cell)
    
    return closest_dir


def turns_until_night(turn_number: int)-> int:
    """
    Gets the number of turns until night. 
    
    Transfer to base 40, then take 30 - x if x < 30, else 0.
    """
    base_number = turn_number % 40
    if base_number == 0:
        return 30  
    elif base_number < 30:
        return 30 - base_number
    else:
        return 0
    

def get_resource_map(square, game_state):
    """
    This will return the map for a single resource tile of the resource density 
    
    """
    resource_map = defaultdict(lambda: defaultdict(int))
    height = game_state.map.height
    
    fuel_amount = square.resource.amount * GAME_CONSTANTS["PARAMETERS"]["RESOURCE_TO_FUEL_RATE"][square.resource.type.upper()]
    
    # Map is always square.
    for i in range(0, height):
        for j in range(0, height):
            resource_map[i][j] = fuel_amount / max(Position(i, j).distance_to(square.pos), 0.5)
    
    return resource_map


def resource_density_by_position(pos, game_state, player):
    """
    Gets a map of the resource closeness of each square divided by how far away from
    the current position they are.
    """
    game_map = game_state.map.map
    height = game_state.map.height
    
    researched_coal = player.researched_coal()
    researched_uranium = player.researched_uranium()
    
    resource_squares = find_resources(game_state, researched_coal=researched_coal, researched_uranium=researched_uranium)
    
    resource_maps = []
    for square in resource_squares:
        resource_map = get_resource_map(square, game_state)
        resource_maps.append(resource_map)
    
    final_map = defaultdict(lambda: defaultdict(int))
    
    for i in range(0, height):
        for j in range(0, height):
            
            distance_to_cell = pos.distance_to(Position(i, j))
            if distance_to_cell > 2 * 5:
                final_map[i][j] = 0
            else:
                final_map[i][j] = sum(resource_map[i][j] for resource_map in resource_maps) / max(0.5, 2 * distance_to_cell)
    
    return final_map


class Mission(ABC):
    staying_still = False
    finished = False
    abandoned = False
    mission_name = None
    current_tile = None
    
    @abstractmethod
    def get_action(self, unit, game_state, player, destination_squares):
        pass
    
    def abandon_mission(self):
        self.abandoned = True
        
    def __str__(self):
        return self.mission_name
    

class HarvestResources(Mission):
    
    target_tile = None
    mission_name = "harvest"
    
    def __init__(self, target_tile, expected_resources, number_of_trips, target_city):
        self.target_tile = target_tile
        self.expected_resources_per_trip = expected_resources
        self.number_of_trips = number_of_trips
        self.target_city = target_city
    
    def get_action(self, unit, game_state, player, destination_squares):
        # we want to mine only if there is space left in the worker's cargo
        if unit.get_cargo_space_left() > 0:
            # create a move action to move this unit in the direction of the target tile and add to our actions list
            direction_to_target_tile = direction_to(unit.pos, self.target_tile.pos, destination_squares, game_state)
            action = unit.move(direction_to_target_tile)
        else:    
            # find the closest citytile and move the unit towards it to drop resources to a citytile to fuel the city
            closest_city_tile = find_closest_city_tile(unit.pos, player, self.target_city)
            if closest_city_tile is not None:
                # create a move action to move this unit in the direction of the closest resource tile and add to our actions list
                direction_to_go = direction_to(unit.pos, closest_city_tile.pos, destination_squares, game_state)
                action = unit.move(direction_to_go)
                if unit.pos.translate(direction_to_go, 1) == closest_city_tile.pos:
                    self.finished = True
                    fuel_received_this_day[closest_city_tile.cityid] -= unit.cargo.wood * 1 + unit.cargo.coal * 10 + unit.cargo.uranium * 40
                
        return action
    
    def get_expected_resources(self) -> int:
        """
        This is probably an underestimate at the moment of the number of resources
        because it doesn't account for adjacency bonuses.
        """
        return self.expected_resources_per_trip * self.number_of_trips
    

class BuildCity(Mission):
    
    mission_name = "build"
    sub_mission = None
    target_city = None
    
    
    def __init__(self, target_city=None):
 
        self.target_city = target_city
    
    
    def get_action(self, unit, game_state, player, destination_squares):
        if unit.get_cargo_space_left() > 0:
            closest_resources_tile = find_closest_resources(unit.pos, player, game_state)
            citytiles = get_all_city_tiles(player)
            direction_to_resources_square = direction_to(unit.pos, closest_resources_tile.pos, destination_squares, game_state, avoid_cities=True, citytiles=citytiles)
            action = unit.move(direction_to_resources_square)  
        
        else:
            if self.target_city is not None:
                best_cell_to_build = find_best_place_to_expand_city(self.target_city, player, game_state, unit)
            else:
                best_cell_to_build = find_best_place_to_expand_city(find_closest_city_tile(unit.pos, player).cityid, player, game_state, unit)
            if unit.pos == best_cell_to_build.pos and unit.can_build(game_state.map):
                action = unit.build_city()
                self.finished = True
            else:
                # Avoid city - otherwise you will get rid of the resources!
                citytiles = get_all_city_tiles(player)
                direction_to_building_square = direction_to(unit.pos, best_cell_to_build.pos, destination_squares, game_state, avoid_cities=True, citytiles=citytiles)
                action = unit.move(direction_to_building_square)  
        return action


class BuildNewCity(Mission):
    
    mission_name = "build_new_city"
    
    def __init__(self, target_tile=None):
        self.target_tile = target_tile
    
    def get_action(self, unit, game_state, player, destination_squares):
        if unit.get_cargo_space_left() > 0:
            closest_resources_tile = find_closest_resources(unit.pos, player, game_state)
            citytiles = get_all_city_tiles(player)
            direction_to_resources_square = direction_to(unit.pos, closest_resources_tile.pos, destination_squares, game_state, avoid_cities=True, citytiles=citytiles)
            action = unit.move(direction_to_resources_square)  

        else:
            if unit.pos == self.target_tile.pos and unit.can_build(game_state.map):
                action = unit.build_city()
                self.finished = True
            else:
                # Avoid city - otherwise you will get rid of the resources!
                citytiles = get_all_city_tiles(player)
                direction_to_building_square = direction_to(unit.pos, self.target_tile.pos, destination_squares, game_state, avoid_cities=True, citytiles=citytiles)
                action = unit.move(direction_to_building_square)  
        return action
    
    
    
class ReturnHome(Mission):
    
    target_city_tile = None
    mission_name = "return_home"
    
    def __init__(self, closest_city_tile):
        self.target_city_tile = closest_city_tile
    
    def get_action(self, unit, game_state, player, destination_squares):
    
        # create a move action to move this unit in the direction of the closest resource tile and add to our actions list
        direction_to_go = direction_to(unit.pos, self.target_city_tile.pos, destination_squares, game_state, is_night=True)
        action = unit.move(direction_to_go)
        if unit.pos.translate(direction_to_go, 1) == self.target_city_tile.pos:
            self.finished = True
        
        return action
        



def get_best_tile_to_harvest(unit, game_state, player, currently_harvested_tiles, turns_til_night):
    collection_rates = {
        'wood': 20,
        'coal': 5,
        'uranium': 2,
    }
    
    fuel_values = {
        'wood': 1,
        'coal': 10,
        'uranium': 40
    }
    cooldown = 2 if unit.is_worker() else 3
    
    resource_tiles_in_range = find_resources_in_range(game_state, unit, turns_til_night / 2, researched_coal=player.researched_coal(), researched_uranium=player.researched_uranium())
    resource_tiles_in_range = [tile for tile in resource_tiles_in_range if tile[0] not in currently_harvested_tiles]
    
    best_tile_to_mine = None
    best_average_amount = 0
    total_mined = 0
    number_of_trips = 1
    
    for tile, distance in resource_tiles_in_range:
        turns_required = cooldown * distance
        potential_turns_remaining = turns_til_night - turns_required
        
        if unit.is_worker():
            turn_cap = 100 / collection_rates[tile.resource.type]
        elif unit.is_cart():
            turn_cap = 2000 / collection_rates[tile.resource.type]
        turns_harvesting = min(turn_cap, potential_turns_remaining)
        
        fuel_harvested = turns_harvesting * fuel_values[tile.resource.type] * collection_rates[tile.resource.type]
        fuel_harvested_per_turn = fuel_harvested / (turns_required + turns_harvesting)
        if fuel_harvested_per_turn > best_average_amount:
            best_average_amount = fuel_harvested_per_turn
            best_tile_to_mine = tile
            total_mined = fuel_harvested
            number_of_trips = int(turns_til_night / (turns_required + turns_harvesting))
    
    return (best_tile_to_mine, total_mined, number_of_trips)


def get_current_fuel_per_city(player: Player) -> defaultdict:
    """
    Returns a defaultdict with the current fuel in each city.
    """
    output = defaultdict(int)
    for city_id, city in player.cities.items():
        output[city_id] = city.fuel
    return output


def get_current_upkeep_state(expected_income, current_resources, fuel_received_this_day, current_upkeep) -> defaultdict:
    """
    This will do the accounting as to how the cities are tracking to survive.
    """
    all_sets = [expected_income, current_resources, fuel_received_this_day, current_upkeep]
    # This will return a set with all of the keys from the union of the above sets.
    all_keys = set().union(*all_sets)
        
    result = {key: sum(d[key] for d in all_sets) for key in all_keys}
    return result


def get_best_place_to_build_new_city(unit, game_state, player, turns_til_night, non_adjacent) -> Position:
    """
    Start by doing something a bit stupid. Just go somewhere that is close, uninhabited, & near a bunch of resources.
    """
    
    resource_map = resource_density_by_position(unit.pos, game_state, player)
    
    not_finished = True
    
    while not_finished:
        tile_with_best_resources = max([(i, j, resource_map[i][j]) for i in resource_map.keys() for j in resource_map.keys()], key=lambda x: x[2])
        # Slightly unsure that this is giving me the right thing. Check behaviour.
        cell = game_state.map.get_cell(tile_with_best_resources[0], tile_with_best_resources[1])
        if cell.resource is None and cell.citytile is None and cell.road == 0:
            if non_adjacent and all([game_state.map.get_cell_by_pos(tile).citytile is None for tile in [cell.pos.translate(DIRECTIONS.NORTH, 1), cell.pos.translate(DIRECTIONS.EAST, 1), cell.pos.translate(DIRECTIONS.SOUTH, 1), cell.pos.translate(DIRECTIONS.WEST, 1)]]):
                return cell
            else:
                return cell
        resource_map[tile_with_best_resources[0]][tile_with_best_resources[1]] = 0


def assign_new_mission(unit, game_state, player, unit_missions, turns_til_night, fuel_received_this_day) -> Mission:
    """
    This is where the brain dishes out the missions.
    
    
    """
    #print("Trying to assign missions", file=sys.stderr)
    current_missions = [v.mission_name for k,v in unit_missions.items()]
    currently_harvested_tiles = [v.target_tile for k,v in unit_missions.items() if v.mission_name == 'harvest']
    
    expected_income = get_expected_resources(unit_missions)
    current_resources = get_current_fuel_per_city(player)
    # It will return upkeep per turn, and we need to keep it alive for 10 turns throughout the night.
    current_upkeep = get_current_upkeep(player)
    
    current_state = get_current_upkeep_state(expected_income, current_resources, fuel_received_this_day, current_upkeep)
    #print(current_state, file=sys.stderr)
    city_meeting_needs = [value >= 0 for key, value in current_state.items()]
    
    #print(player.cities, file=sys.stderr)
    
    if not all(city_meeting_needs):
        #print("Not meeting needs", file=sys.stderr)
        city_who_needs_more_resources = min(current_state, key=current_state.get)
        target_tile, total_mined, number_of_trips = get_best_tile_to_harvest(unit, game_state, player, currently_harvested_tiles, turns_til_night)
        if target_tile is not None:
            return HarvestResources(target_tile, total_mined, number_of_trips, city_who_needs_more_resources)
    
    if all(city_meeting_needs) and max(current_state.values()) >= 150:
        #print("Building", file=sys.stderr)
        someone_already_building = 'build' in current_missions or 'build_new_city' in current_missions
        best_city = max(current_state, key=current_state.get)
        if len(player.units) >= get_number_of_city_tiles(player) and not someone_already_building and unit.is_worker():
            if len(player.cities.keys()) < 2:
                # If there is more than five tiles in the city then say we need a new city.
                non_adjacent = len(player.cities[list(player.cities.keys())[0]].citytiles) > 5
                target_tile = get_best_place_to_build_new_city(unit, game_state, player, turns_til_night, non_adjacent=non_adjacent)
                return BuildNewCity(target_tile)
            else:
                return BuildCity(best_city)
        else:
            target_tile, total_mined, number_of_trips = get_best_tile_to_harvest(unit, game_state, player, currently_harvested_tiles, turns_til_night)
            city_who_needs_more_resources = min(current_state, key=current_state.get)
            if target_tile is not None:
                return HarvestResources(target_tile, total_mined, number_of_trips, city_who_needs_more_resources)
            else:
                closest_city_tile = find_closest_city_tile(unit.pos, player)
                return ReturnHome(closest_city_tile)
    else:
        #print("Other", file=sys.stderr)
        target_tile, total_mined, number_of_trips = get_best_tile_to_harvest(unit, game_state, player, currently_harvested_tiles, turns_til_night)
        city_who_needs_more_resources = min(current_state, key=current_state.get)
        if target_tile is not None:
            return HarvestResources(target_tile, total_mined, number_of_trips, city_who_needs_more_resources)
        else:
            closest_city_tile = find_closest_city_tile(unit.pos, player)
            return ReturnHome(closest_city_tile)
        
        
def get_current_worker_cart_split(player: Player) -> float:
    """
    Gets the current % of workers to carts.
    """
    num_workers = 0
    num_carts = 0
    for unit in player.units:
        if unit.is_worker():
            num_workers += 1
        else:
            num_carts += 1
        
    return 100 * num_workers / (num_workers + num_carts)


def get_expected_resources(unit_missions) -> defaultdict:
    """
    This is currently assuming that everyone just goes once - doesn't account for multiple trips.
    """
    output = defaultdict(int)
    for unit_id, mission in unit_missions.items():
        if mission.mission_name == 'harvest':
            output[mission.target_city] += mission.get_expected_resources()
    return output


def get_current_upkeep(player) -> defaultdict:
    """
    This will get the current light upkeep of the cities.
    We want this to be negative as it will be offsetting against positive income values later
    and want to be able to just sum everything.
    """
    output = defaultdict(int)
    for city_id, city in player.cities.items():
        output[city_id] = -10 * city.get_light_upkeep()
    return output

# we declare this global game_state object so that state persists across turns so we do not need to reinitialize it all the time
game_state = None
unit_missions = {}
target_worker_cart_split = 100
fuel_received_this_day = defaultdict(int)


def agent(observation, configuration):
    global game_state
    global unit_missions
    global target_worker_cart_split
    global fuel_received_this_day

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])
    
    actions = []
        
    ### AI Code goes down here! ### 
    player = game_state.players[observation.player]
    opponent = game_state.players[(observation.player + 1) % 2]
    width, height = game_state.map.width, game_state.map.height
    
    # add debug statements like so!
      
    destination_squares = []
    
    turns_til_night = turns_until_night(observation['step'])
    # If it's the start of the day phase then reset the amount of fuel received this day.
    if turns_til_night == 30:
        fuel_received_this_day = defaultdict(int)
    
   # print(turns_til_night, file=sys.stderr)
    unit_missions = {k: v for k,v in unit_missions.items() if not v.finished or not v.abandoned}
    if turns_til_night == 30:
        unit_missions = {k: v for k,v in unit_missions.items() if v.mission_name != 'return_home'}
    units_to_act = []
    
    #print(unit_missions, file=sys.stderr)
    
    for unit in player.units:
        if not unit.can_act():
            destination_squares.append(unit.pos)
        else:
            units_to_act.append(unit)
        
        
    for unit in units_to_act:
        cooldown = 2 if unit.is_worker() else 3
        action = None
        # Check if they need to start heading back home. This is horribly inefficient but I guess works unless we run into time issues?
        
        if unit.id in unit_missions.keys() and hasattr(unit_missions[unit.id], 'target_city'):
            closest_city_tile = find_closest_city_tile(unit.pos, player, unit_missions[unit.id].target_city)
        else:
            closest_city_tile = find_closest_city_tile(unit.pos, player)
        if closest_city_tile is not None and unit.pos.distance_to(closest_city_tile.pos) * cooldown + 1 >= turns_til_night:
            unit_missions[unit.id] = ReturnHome(closest_city_tile)
            
        # If they have a mission, then get the next action for their mission.
        # If they're in this bucket they must be able to act, so don't need to check anymore.
        if unit.id in unit_missions.keys():
            mission = unit_missions[unit.id]
            action = mission.get_action(unit, game_state, player, destination_squares)
            
        else:
            # There must not be a mission for this unit.
            mission = assign_new_mission(unit, game_state, player, unit_missions, turns_til_night, fuel_received_this_day)
            if mission is not None:
                unit_missions[unit.id] = mission
                action = mission.get_action(unit, game_state, player, destination_squares)
            else:
                mission = None
        
        if action is not None:
            actions.append(action)

    # Now go through the city tiles.
    citytiles = get_all_city_tiles(player)
    
    tile_already_producing = False
    for citytile in citytiles:
        if citytile.can_act():
            # If you can produce and another tile has not yet produced, then produce a worker
            if not tile_already_producing and len(citytiles) > len(player.units):
                current_split = get_current_worker_cart_split(player) 
                #print(f'Current split is: {current_split}', file=sys.stderr)
                if current_split < target_worker_cart_split:
                    #print("Building worker", file=sys.stderr)
                    action = citytile.build_worker()
                    
                else:
                    #action = citytile.build_cart()
                    action = citytile.build_worker()
                tile_already_producing = True
            else:
                 action = citytile.research()
            
            actions.append(action)
    #print(actions, file=sys.stderr)
    #print(unit_missions, file=sys.stderr)
    return actions

