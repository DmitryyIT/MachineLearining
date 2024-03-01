{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMP23/3p7MckncQ4I86fws3",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/DmitryyIT/MachineLearining/blob/main/LinearRegression.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "VMyCoUdNAByE"
      },
      "outputs": [],
      "source": [
        "prices = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]\n",
        "demand = [180, 170, 150, 140, 135, 128, 120, 115, 114, 108, 100, 97, 96, 95, 90, 85, 80, 80, 80]"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "\n",
        "prices = np.array(prices).reshape(-1, 1)\n",
        "demand = np.array(demand)"
      ],
      "metadata": {
        "id": "HDreRX7RCwGI"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(prices, demand, test_size = 0.2, random_state = 42)"
      ],
      "metadata": {
        "id": "x2pbQfC_CwUQ"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)"
      ],
      "metadata": {
        "id": "d0WybA5XC1Kq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.linear_model import LinearRegression\n",
        "\n",
        "model = LinearRegression()\n",
        "\n",
        "model.fit(X_train, y_train)"
      ],
      "metadata": {
        "id": "uT5beeW1C2VU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(model.coef_, model.intercept_)"
      ],
      "metadata": {
        "id": "fzCg4en5C9qh"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred = model.predict(X_test)\n",
        "\n",
        "y_pred[:3]"
      ],
      "metadata": {
        "id": "SEhyOqp_C_x-"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn import metrics\n",
        "\n",
        "print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))"
      ],
      "metadata": {
        "id": "YNDSsraYC_8X"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}