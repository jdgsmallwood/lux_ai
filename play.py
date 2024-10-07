from kaggle_environments import make

# Important imports
from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES, Position
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux.game_objects import Player, City, Unit, CityTile
from lux import annotate
from queue import PriorityQueue
import random
import math
import sys

DIRECTIONS = Constants.DIRECTIONS


# env = make("lux_ai_2021", configuration={"seed": 562124210, "loglevel": 2, "annotations": True}, debug=True)

# this snippet finds all resources stored on the map and puts them into a list so we can search over them
def find_resources(game_state, researched_coal=False, researched_uranium=False):
    resource_tiles: list[Cell] = []
    width, height = game_state.map_width, game_state.map_height
    for y in range(height):
        for x in range(width):
            cell = game_state.map.get_cell(x, y)
            if cell.has_resource():
                if cell.resource.type == Constants.RESOURCE_TYPES.COAL and not researched_coal: continue
                if cell.resource.type == Constants.RESOURCE_TYPES.URANIUM and not researched_uranium: continue
                resource_tiles.append(cell)
    return resource_tiles


def find_resources_in_range(game_state, unit, unit_range, researched_coal=False, researched_uranium=False) -> list[
    (Cell, int)]:
    resource_tiles = find_resources(game_state, researched_coal, researched_uranium)
    resource_tiles_in_range = [(tile, unit.pos.distance_to(tile.pos)) for tile in resource_tiles if
                               unit.pos.distance_to(tile.pos) < unit_range]
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


# snippet to find the closest city tile to a position
def find_closest_city_tile(pos, player):
    closest_city_tile = None
    if len(player.cities) > 0:
        closest_dist = math.inf
        # the cities are stored as a dictionary mapping city id to the city object, which has a citytiles field that
        # contains the information of all citytiles in that city
        for k, city in player.cities.items():
            for city_tile in city.citytiles:
                dist = city_tile.pos.distance_to(pos)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_city_tile = city_tile
    return closest_city_tile


def get_number_of_city_tiles(player: Player):
    return len(get_all_city_tiles(player))


def get_all_city_tiles(player: Player) -> list[CityTile]:
    """
    This will return a list of all city tiles for a particular player.
    """
    output = []
    for id, city in player.cities.items():
        output += city.citytiles
    return output


# This will get the adjacent squares to a particular tile
def get_adjacent_cells(pos: Position, game_state) -> list[Cell]:
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
def get_adjacent_cells_to_build(pos: Position, game_state) -> list[Cell]:
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
    adjacent_tiles = [v for city_tile in city_tiles for v in get_adjacent_cells_to_build(city_tile.pos, game_state)]

    closest_tile = None
    closest_distance = 1000

    for tile in adjacent_tiles:
        dis = unit.pos.distance_to(tile.pos)
        if dis < closest_distance:
            closest_distance = dis
            closest_tile = tile
    return closest_tile


class Node():

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


def direction_to(start_pos: Position, end_pos: Position, already_used_squares: list[Position], game_state,
                 avoid_cities=False, is_night=False, citytiles=None) -> DIRECTIONS:
    """
    Return closest position to end_pos. If avoid_cities is True then it will go around cities.

    Collision avoidance - don't go to a square that's already been claimed by someone else.
    """
    closest_dist = start_pos.distance_to(end_pos)

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

    open_list = PriorityQueue()
    open_list.put(initial_square)
    closed_list = []

    final_path = None

    while not open_list.empty():

        # Get the lowest f-value on the current list.
        current_node = open_list.get()
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

            # open_nodes_same_pos = [node for priority, node in open_list if node.position == child.position]
            # if len(open_nodes_same_pos) > 0 and child.g > open_nodes_same_pos[0].g:
            #    continue

            open_list.put(child)

    new_cell = final_path[1]
    new_cell_obj = game_state.map.get_cell_by_pos(new_cell)
    if new_cell_obj.citytile is None:
        already_used_squares.append(new_cell)
    closest_dir = start_pos.direction_to(new_cell)
    return closest_dir


def turns_until_night(turn_number: int) -> int:
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


# def direction_to(start_pos: Position, end_pos: Position, already_used_squares: list[Position], game_state, avoid_cities=False, is_night=False, citytiles=None) -> DIRECTIONS:
#     """
#     Return closest position to end_pos. If avoid_cities is True then it will go around cities.

#     Collision avoidance - don't go to a square that's already been claimed by someone else.
#     """
#     check_dirs = [
#             DIRECTIONS.NORTH,
#             DIRECTIONS.EAST,
#             DIRECTIONS.SOUTH,
#             DIRECTIONS.WEST,
#         ]
#     closest_dist = start_pos.distance_to(end_pos)
#     closest_dir = DIRECTIONS.CENTER

#     if not start_pos == end_pos:

#         # This is a list to keep track of squares that are as good as current fit.
#         as_good_as = []
#         for direction in check_dirs:
#             newpos = start_pos.translate(direction, 1)
#             if newpos not in already_used_squares:
#                 try:
#                     cell = game_state.map.get_cell(newpos.x, newpos.y)
#                     if avoid_cities and cell.citytile is not None:
#                         continue
#                     dist = end_pos.distance_to(newpos)
#                     if dist < closest_dist:
#                         closest_dir = direction
#                         closest_dist = dist
#                 except:
#                     pass

#         # This will only happen if they are somehow surrounded by city, in which
#         # case just go anywhere.
#         if not is_night and closest_dir == DIRECTIONS.CENTER:
#             if len(as_good_as) > 0:
#                 closest_dir = random.choice(as_good_as)
#             else:
#                 random_direction_with_no_other_unit = False
#                 while not random_direction_with_no_other_unit:
#                     random_direction = random.choice(check_dirs + [DIRECTIONS.CENTER])
#                     new_cell = start_pos.translate(random_direction, 1)

#                     if new_cell not in already_used_squares:
#                         if not avoid_cities:
#                             random_direction_with_no_other_unit = True
#                             closest_dir = random_direction
#                         else:
#                             try:
#                                 cell = game_state.map.get_cell(new_cell.x, new_cell.y)
#                                 if cell.citytile is None:
#                                     random_direction_with_no_other_unit = True
#                                     closest_dir = random_direction
#                             except:
#                                 pass

#     new_cell = start_pos.translate(closest_dir, 1)
#     new_cell_obj = game_state.map.get_cell_by_pos(new_cell)
#     if new_cell_obj.citytile is None:
#         already_used_squares.append(new_cell)

#     return closest_dir

from abc import ABC, abstractmethod


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

    def __init__(self, target_tile, expected_resources, number_of_trips):
        self.target_tile = target_tile
        self.expected_resources_per_trip = expected_resources
        self.number_of_trips = number_of_trips

    def get_action(self, unit, game_state, player, destination_squares):
        # we want to mine only if there is space left in the worker's cargo
        if unit.get_cargo_space_left() > 0:
            # create a move action to move this unit in the direction of the target tile and add to our actions list
            direction_to_target_tile = direction_to(unit.pos, self.target_tile.pos, destination_squares, game_state)
            action = unit.move(direction_to_target_tile)
        else:
            # find the closest citytile and move the unit towards it to drop resources to a citytile to fuel the city
            closest_city_tile = find_closest_city_tile(unit.pos, player)
            if closest_city_tile is not None:
                # create a move action to move this unit in the direction of the closest resource tile and add to our actions list
                direction_to_go = direction_to(unit.pos, closest_city_tile.pos, destination_squares, game_state)
                action = unit.move(direction_to_go)
                if unit.pos.translate(direction_to_go, 1) == self.target_tile.pos:
                    self.finished = True
        return action

    def get_expected_resources(self):
        return self.expected_resources_per_trip * self.number_of_trips


class BuildCity(Mission):
    mission_name = "build"
    sub_mission = None

    def get_action(self, unit, game_state, player, destination_squares):
        if unit.get_cargo_space_left() > 0:
            closest_resources_tile = find_closest_resources(unit.pos, player, game_state)
            citytiles = get_all_city_tiles(player)
            direction_to_resources_square = direction_to(unit.pos, closest_resources_tile.pos, destination_squares,
                                                         game_state, avoid_cities=True, citytiles=citytiles)
            action = unit.move(direction_to_resources_square)

        else:
            best_cell_to_build = find_best_place_to_expand_city(find_closest_city_tile(unit.pos, player).cityid, player,
                                                                game_state, unit)
            if unit.pos == best_cell_to_build.pos and unit.can_build(game_state.map):
                action = unit.build_city()
                self.finished = True
            else:
                # Avoid city - otherwise you will get rid of the resources!
                citytiles = get_all_city_tiles(player)
                direction_to_building_square = direction_to(unit.pos, best_cell_to_build.pos, destination_squares,
                                                            game_state, avoid_cities=True, citytiles=citytiles)
                action = unit.move(direction_to_building_square)
        return action


class ReturnHome(Mission):
    target_city_tile = None
    mission_name = "return_home"

    def __init__(self, closest_city_tile):
        self.target_city_tile = closest_city_tile

    def get_action(self, unit, game_state, player, destination_squares):
        # create a move action to move this unit in the direction of the closest resource tile and add to our actions list
        direction_to_go = direction_to(unit.pos, self.target_city_tile.pos, destination_squares, game_state,
                                       is_night=True)
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

    resource_tiles_in_range = find_resources_in_range(game_state, unit, turns_til_night / 2,
                                                      researched_coal=player.researched_coal(),
                                                      researched_uranium=player.researched_uranium())
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


def assign_new_mission(unit, game_state, player, unit_missions, turns_til_night) -> Mission:
    current_missions = [v.mission_name for k, v in unit_missions.items()]
    currently_harvested_tiles = [v.target_tile for k, v in unit_missions.items() if v.mission_name == 'harvest']

    someone_already_building = 'build' in current_missions

    if len(player.units) == get_number_of_city_tiles(player) and not someone_already_building and unit.is_worker():
        return BuildCity()
    else:
        target_tile, total_mined, number_of_trips = get_best_tile_to_harvest(unit, game_state, player,
                                                                             currently_harvested_tiles, turns_til_night)
        if target_tile is not None:
            return HarvestResources(target_tile, total_mined, number_of_trips)
        else:
            return None


def get_current_worker_cart_split(player: Player) -> float:
    """
    Gets the current %
    """
    num_workers = 0
    num_carts = 0
    for unit in player.units:
        if unit.is_worker():
            num_workers += 1
        else:
            num_carts += 1

    return 100 * num_workers / (num_workers + num_carts)


def get_expected_resources(unit_missions) -> int:
    """
    This is currently assuming that everyone just goes once - doesn't account for multiple trips.
    """
    sum = 0
    return sum


# we declare this global game_state object so that state persists across turns so we do not need to reinitialize it all the time
game_state = None
unit_missions = {}
target_worker_cart_split = 66


def agent(observation, configuration):
    print(observation['remainingOverageTime'], file=sys.stderr)
    global game_state
    global unit_missions
    global target_worker_cart_split

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])

    actions = []
    print("NEW TURN", file=sys.stderr)

    ### AI Code goes down here! ###
    player = game_state.players[observation.player]
    opponent = game_state.players[(observation.player + 1) % 2]
    width, height = game_state.map.width, game_state.map.height

    # add debug statements like so!

    destination_squares = []

    turns_til_night = turns_until_night(observation['step'])
    print(turns_til_night, file=sys.stderr)
    unit_missions = {k: v for k, v in unit_missions.items() if not v.finished or not v.abandoned}
    if turns_til_night == 30:
        unit_missions = {k: v for k, v in unit_missions.items() if v.mission_name != 'return_home'}
    units_to_act = []

    # print(unit_missions, file=sys.stderr)

    for unit in player.units:
        if not unit.can_act():
            destination_squares.append(unit.pos)
        else:
            units_to_act.append(unit)

    for unit in units_to_act:
        cooldown = 2 if unit.is_worker() else 3

        # Check if they need to start heading back home. This is horribly inefficient but I guess works unless we run into time issues?
        closest_city_tile = find_closest_city_tile(unit.pos, player)
        if closest_city_tile is not None and unit.pos.distance_to(
                closest_city_tile.pos) * cooldown + 1 >= turns_til_night:
            unit_missions[unit.id] = ReturnHome(closest_city_tile)

        # If they have a mission, then get the next action for their mission.
        # If they're in this bucket they must be able to act, so don't need to check anymore.
        if unit.id in unit_missions.keys():
            mission = unit_missions[unit.id]
            action = mission.get_action(unit, game_state, player, destination_squares)

        else:
            # There must not be a mission for this unit.
            mission = assign_new_mission(unit, game_state, player, unit_missions, turns_til_night)
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
                print(f'Current split is: {current_split}', file=sys.stderr)
                if current_split < target_worker_cart_split:
                    print("Building worker", file=sys.stderr)
                    action = citytile.build_worker()
                    # action = citytile.research()
                else:
                    # action = citytile.build_cart()
                    action = citytile.build_worker()
                    # action = citytile.research()
                tile_already_producing = True
            else:
                action = citytile.research()

            actions.append(action)
    print(actions, file=sys.stderr)
    print(unit_missions, file=sys.stderr)
    return actions


env = make("lux_ai_2021", configuration={"seed": 562124215, "loglevel": 2, "annotations": True}, debug=True)
steps = env.run([agent, "simple_agent"])
#env.render(mode="ipython", width=1200, height=800)