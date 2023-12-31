# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from aerial_gym.envs.base.aerial_robot_config import AerialRobotCfg
from aerial_gym.envs.base.aerial_robot_with_obstacles_config import AerialRobotWithObstaclesCfg
from aerial_gym.envs.base.aerial_robot_with_trees_config import AerialRobotWithTreesCfg
from aerial_gym.envs.base.aerial_robot_with_vision_config import AerialRobotWithVisionCfg
from aerial_gym.envs.base.aerial_robot_with_vision_testing_config import AerialRobotWithVisionTestCfg
from .base.aerial_robot  import AerialRobot
from .base.aerial_robot_with_obstacles import AerialRobotWithObstacles
from .base.aerial_robot_with_trees import AerialRobotWithTrees
from .base.aerial_robot_with_vision import AerialRobotWithVision
from .base.aerial_robot_with_vision_testing import AerialRobotWithVisionTest

import os

from aerial_gym.utils.task_registry import task_registry

task_registry.register("quad", AerialRobot, AerialRobotCfg())
task_registry.register("quad_with_obstacles", AerialRobotWithObstacles, AerialRobotWithObstaclesCfg())
task_registry.register("quad_with_trees", AerialRobotWithTrees, AerialRobotWithTreesCfg())
task_registry.register("quad_with_vision", AerialRobotWithVision, AerialRobotWithVisionCfg())
task_registry.register("quad_with_vision_testing", AerialRobotWithVisionTest, AerialRobotWithVisionTestCfg())