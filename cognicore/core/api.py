"""
CogniCore Core API — Simple entry points for training and evaluation.
"""

from typing import Any
import logging

logger = logging.getLogger("cognicore")

def train(agent: Any, env: Any, episodes: int = 10) -> Any:
    """Train an agent in a CogniCore environment.
    
    Parameters
    ----------
    agent : Any
        An agent implementing `act()`, `on_reward()`, and `on_episode_end()`.
    env : CogniCoreEnv
        The environment instance.
    episodes : int
        Number of episodes to run.
        
    Returns
    -------
    Any
        The trained agent.
    """
    logger.info(f"Started training agent for {episodes} episodes.")
    
    for ep in range(episodes):
        obs = env.reset()
        while True:
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            
            if hasattr(agent, "on_reward"):
                agent.on_reward(reward)
                
            if done:
                if hasattr(agent, "on_episode_end"):
                    agent.on_episode_end(env.episode_stats())
                break
                
    logger.info("Training complete.")
    return agent

def evaluate(agent: Any, env: Any, episodes: int = 5) -> float:
    """Evaluate an agent's performance in a CogniCore environment.
    
    Parameters
    ----------
    agent : Any
        An agent implementing `act()`.
    env : CogniCoreEnv
        The environment instance.
    episodes : int
        Number of episodes to evaluate.
        
    Returns
    -------
    float
        The average score (0.0 to 1.0) across all episodes.
    """
    logger.info(f"Started evaluation for {episodes} episodes.")
    total_score = 0.0
    
    for ep in range(episodes):
        obs = env.reset()
        while True:
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            if done:
                total_score += env.get_score()
                break
                
    final_score = total_score / episodes if episodes > 0 else 0.0
    logger.info(f"Evaluation complete. Average score: {final_score:.4f}")
    return final_score
