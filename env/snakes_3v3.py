from .snakes import SnakeEatBeans
from copy import deepcopy

class Snake3V3(SnakeEatBeans):
    def __init__(self, conf):
        super().__init__(conf)
    

    def get_next_state(self, all_action):
        before_info = self.step_before_info()
        not_valid = self.is_not_valid_action(all_action)
        self.snakes_before = deepcopy(self.players)
        if not not_valid:
            # 各玩家行动
            # print("current_state", self.current_state)
            eat_snakes = [0] * self.n_player

            self.info_after = {
                'eat_bean':[],  # a(0) eat at pos(1)
                'hit':[],    # a(0) eat b(1)
                'win': -1  # -1 undone, 0: team0 , 1: team1
            }

            # move and eat beans
            for i in range(self.n_player):
                snake = self.players[i]
                act = self.actions[all_action[i][0].index(1)]
                # print(snake.player_id, "此轮的动作为：", self.actions_name[act])
                snake.change_direction(act)
                snake.move_and_add(self.snakes_position)
                # logic of eat beans
                if self.be_eaten(snake.headPos):  # @yanxue
                    # snake.snake_reward = 1
                    eat_snakes[i] = 1
                    self.info_after["eat_bean"].append((i, snake.headPos))
                else:
                    # snake.snake_reward = 0
                    snake.pop()
                # print(snake.player_id, snake.segments)   # @yanxue


            snake_position = [[-1] * self.board_width for _ in range(self.board_height)]
            re_generatelist = [0] * self.n_player

            # snakes hit
            for i in range(self.n_player):
                snake = self.players[i]
                segment = snake.segments
                for j in range(len(segment)):

                    x = segment[j][0]
                    y = segment[j][1]
                    if snake_position[x][y] != -1:
                        if j == 0:  # 撞头
                            re_generatelist[i] = 1
                            self.info_after["hit"].append((snake_position[x][y], i))
                        compare_snake = self.players[snake_position[x][y]]
                        if [x, y] == compare_snake.segments[0]:  # 两头相撞
                            re_generatelist[snake_position[x][y]] = 1
                            self.info_after["hit"].append((i, snake_position[x][y]))
                    else:
                        snake_position[x][y] = i

            # reward assignment for hit unit
            for i in range(self.n_player):
                snake = self.players[i]
                if re_generatelist[i] == 1:
                    if eat_snakes[i] == 1:
                        snake.snake_reward = self.init_len - len(snake.segments) + 1
                    else:
                        snake.snake_reward = self.init_len - len(snake.segments)
                    snake.segments = []


            for i in range(self.n_player):
                snake = self.players[i]
                if re_generatelist[i] == 1:
                    snake = self.clear_or_regenerate(snake)
                self.snakes_position[snake.player_id] = snake.segments
                snake.score = snake.get_score()
            # yanxue add
            # 更新状态
            self.generate_beans()

            next_state = self.update_state()
            self.current_state = next_state
            self.step_cnt += 1

            self.won = [0] * self.n_player

            for i in range(self.n_player):
                s = self.players[i]
                self.won[i] = s.score

            # self.info_after['win'] = 

            self.all_observes = self.get_all_observes(before_info)


            return self.current_state, self.info_after
    
    def get_reward(self):
        # todo: zero sum
        reward_individual = [0] * self.n_player
        reward_team = [0] * 2  #2 teams
        # reward_team

        for i in range(self.n_player):
            reward_individual[i] = len(self.players[i].segments) - len(self.snakes_before[i].segments)

        reward_team[1] = -sum(reward_individual[:self.n_player//2])
        reward_team[0] = -sum(reward_individual[self.n_player//2:])

        r = [0] * self.n_player

        n_team_players = self.n_player // 2


        for i in range(self.n_player):
            if i < n_team_players:
                team_id = 0
            else:
                team_id = 1

            avg_reward = reward_team[team_id] / n_team_players

            r[i] = reward_individual[i] + avg_reward
            self.n_return[i] += r[i]


        # avg_rs = [sum(r[self.n_player//2:])/n_team_players, sum(r[:self.n_player//2])/n_team_players]
        # team_spirit = 0

        # for i in range(self.n_player):
        #     if i < n_team_players:
        #         team_id = 0
        #     else:
        #         team_id = 1


        #     avg_r = avg_rs[team_id]
        #     r[i] = (1 - team_spirit)r[i] + team_spirit * avg_r

        return r

    
    def step(self, joint_action):

        board_state, info_after = self.get_next_state(joint_action)
        done = self.is_terminal()
        reward = self.get_reward()

        return board_state, reward, done, info_after
