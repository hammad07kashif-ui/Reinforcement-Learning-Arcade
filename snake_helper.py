import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores, accuracies):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    
    plt.subplot(2, 1, 1)
    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    plt.ylim(ymin=0)
    if len(scores) > 0:
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
        plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.legend(loc='upper left')

    plt.subplot(2, 1, 2)
    plt.xlabel('Number of Games')
    plt.ylabel('Accuracy (%)')
    plt.plot(accuracies, label='Win Rate (>0 score)', color='green')
    plt.ylim(0, 100)
    if len(accuracies) > 0:
        plt.text(len(accuracies)-1, accuracies[-1], str(round(accuracies[-1], 1)) + "%")
    plt.legend(loc='upper left')

    plt.show(block=False)
    plt.pause(.1)