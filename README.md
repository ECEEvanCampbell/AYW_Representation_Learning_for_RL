# Representation Learning for Reinforcement Learning

Reinforcement learning is challenging when the input space is large. This problem is commonly encountered in visuomotor problems, for instance where a camera is used to guide a robotic hand through a goal-driven task. Traditionally, these tasks rely on a perception model that interprets the approximate location of objects in the image and a control model that computes the robot’s path to complete the goal-driven behaviour.  These two models are trained independently, often relying on models like AlexNet for object perception and training the control model from scratch.  Under this framework, the input space to the control model is typically low dimensional, and lends well to reinforcement learning agents that converge quickly to suitable policies.  Novel models, however, have shifted to using end-to-end training (cite Sergey Levine end-to-end), where the perception and control models are optimized in tandem. Although this strategy has many advantages, the reinforcement learning optimization of end-to-end training now involves much higher input dimension than traditional approaches; this leads to longer training times, and less certain convergence to suitable agents.
Our project investigates a network topology that reduces the input dimension of the reinforcement learning model without relying on standardized perception models. In short, our research question is: “Does representation learning aid in end-to-end reinforcement learning policies”.