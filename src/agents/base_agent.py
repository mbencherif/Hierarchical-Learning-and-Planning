from abc import ABC, abstractmethod


class BaseAgent(ABC):

    """Base Agent. Overload sense and plan modules to add uncertainties."""

    def sense(self, state):
        """
        Create the observation the agent receives from the env's current state.

        This function can be overloaded to simulate deviations of the state
        observation. Typical variations include adding noise or shift to
        simulate noisy sensors or changing sensor-setups.

        :param state: nd.array or torch.FloatTensor
            State the agent receives

        :return observation: nd.array or torch.FloatTensor
            Modified state.
        """
        return state

    @abstractmethod
    def plan(self, observation):
        """
        Choose the next action given the current state's observation.

        :param observation: nd.array or torch.FloatTensor
            Observation of the environment at the current time-step.

        :return action: nd.array or torch.FloatTensor
            The next action to execute.
        """
        raise NotImplementedError

    def act(self, action):
        """
        Modify the action choosen by the agent which will be executed.

        This function can be overloaded to simulate deviations of the action
        execution. Typical variations include adding noise or shift to simulate
        noisy execution of actions or a shift of behavior of the execution
        module.

        :param action: nd.array or torch.FloatTensor
            The action choosen by the agent.

        :return control_action: nd.array or torch.FloatTensor
            The action which is finally executed in the environment.
        """
        return action

    def operate(self, state):
        """
        Execute the complete Sense-Plan-Act pipeline.

        :param state: nd.array or torch.FloatTensor
            State the agent receives

        :return control_action: nd.array or torch.FloatTensor
            The action which is finally executed in the environment.
        """
        obs = self.sense(state)
        action = self.plan(obs)
        control_action = self.act(action)
        return control_action
