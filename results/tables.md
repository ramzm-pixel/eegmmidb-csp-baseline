<!-- transfer -->
| method | real L/R | imagined L/R | real feet | imagined feet |
|---|---|---|---|---|
| within-subject (leave-one-run-out) | 0.639 | 0.626 | 0.770 | 0.676 |
| zero-shot, raw | 0.617 | 0.600 | 0.541 | 0.608 |
| zero-shot, aligned | 0.717 | 0.679 | 0.734 | 0.673 |
| zero-shot, aligned + centred | 0.725 | 0.684 | 0.745 | 0.681 |

<!-- calibration -->
| task | k | from scratch (k trials only) | pooled filters, new LDA | adapt: centre only | adapt: centre + blend means | zero-shot, aligned + centred |
|---|---|---|---|---|---|---|
| real L/R | 2 | 0.523 | 0.550 | 0.703 | 0.689 | 0.725 |
| real L/R | 5 | 0.572 | 0.641 | 0.713 | 0.692 | 0.725 |
| real L/R | 10 | 0.607 | 0.671 | 0.719 | 0.696 | 0.724 |
| imagined L/R | 2 | 0.522 | 0.550 | 0.664 | 0.653 | 0.683 |
| imagined L/R | 5 | 0.559 | 0.615 | 0.674 | 0.650 | 0.683 |
| imagined L/R | 10 | 0.577 | 0.634 | 0.681 | 0.649 | 0.686 |
| real feet | 2 | 0.601 | 0.629 | 0.726 | 0.743 | 0.744 |
| real feet | 5 | 0.711 | 0.749 | 0.736 | 0.765 | 0.743 |
| real feet | 10 | 0.752 | 0.774 | 0.738 | 0.776 | 0.742 |
| imagined feet | 2 | 0.559 | 0.586 | 0.663 | 0.657 | 0.680 |
| imagined feet | 5 | 0.615 | 0.640 | 0.673 | 0.664 | 0.682 |
| imagined feet | 10 | 0.637 | 0.664 | 0.673 | 0.665 | 0.678 |

<!-- preprocessing -->
| factor | task | setting | effect (points) | 95% interval (points) |
|---|---|---|---|---|
| ICA vs no ICA | real L/R | motor channels | -1.20 | ±1.35 |
| ICA vs no ICA | real L/R | hybrid channels | 0.14 | ±1.43 |
| ICA vs no ICA | imagined L/R | motor channels | -0.73 | ±1.01 |
| ICA vs no ICA | imagined L/R | hybrid channels | -1.59 | ±1.49 |
| ICA vs no ICA | real feet | motor channels | -0.67 | ±0.87 |
| ICA vs no ICA | real feet | hybrid channels | 0.56 | ±0.93 |
| ICA vs no ICA | imagined feet | motor channels | 0.11 | ±0.91 |
| ICA vs no ICA | imagined feet | hybrid channels | 0.25 | ±1.31 |
| 21 motor vs 42 channels | real L/R | no ICA | 1.47 | ±1.59 |
| 21 motor vs 42 channels | real L/R | ICA | 0.14 | ±1.64 |
| 21 motor vs 42 channels | imagined L/R | no ICA | 1.11 | ±1.89 |
| 21 motor vs 42 channels | imagined L/R | ICA | 1.97 | ±1.72 |
| 21 motor vs 42 channels | real feet | no ICA | 3.07 | ±1.53 |
| 21 motor vs 42 channels | real feet | ICA | 1.85 | ±1.47 |
| 21 motor vs 42 channels | imagined feet | no ICA | 1.72 | ±1.61 |
| 21 motor vs 42 channels | imagined feet | ICA | 1.59 | ±1.49 |

<!-- subjects -->
| task | method | mean | sd | share above chance |
|---|---|---|---|---|
| imagined feet | zero-shot, aligned + centred | 0.681 | 0.154 | 61% |
| imagined feet | within-subject (leave-one-run-out) | 0.676 | 0.156 | 58% |
| imagined L/R | zero-shot, aligned + centred | 0.684 | 0.156 | 59% |
| imagined L/R | within-subject (leave-one-run-out) | 0.626 | 0.146 | 42% |
| real feet | zero-shot, aligned + centred | 0.745 | 0.175 | 74% |
| real feet | within-subject (leave-one-run-out) | 0.770 | 0.167 | 77% |
| real L/R | zero-shot, aligned + centred | 0.725 | 0.135 | 73% |
| real L/R | within-subject (leave-one-run-out) | 0.639 | 0.141 | 48% |

<!-- timing -->
| window start (s) | real L/R | imagined L/R | real feet | imagined feet |
|---|---|---|---|---|
| 0.000 | 0.599 | 0.601 | 0.663 | 0.612 |
| 0.250 | 0.620 | 0.610 | 0.723 | 0.648 |
| 0.500 | 0.607 | 0.602 | 0.754 | 0.676 |
| 0.750 | 0.620 | 0.586 | 0.762 | 0.677 |
| 1.000 | 0.622 | 0.589 | 0.766 | 0.678 |
| 1.250 | 0.623 | 0.582 | 0.755 | 0.666 |
| 1.500 | 0.622 | 0.582 | 0.744 | 0.651 |
| 1.750 | 0.618 | 0.576 | 0.747 | 0.636 |
| 2.000 | 0.614 | 0.582 | 0.744 | 0.632 |
| 2.250 | 0.613 | 0.574 | 0.740 | 0.628 |
| 2.500 | 0.610 | 0.560 | 0.730 | 0.624 |
| 2.750 | 0.606 | 0.563 | 0.722 | 0.629 |
| 3.000 | 0.593 | 0.568 | 0.710 | 0.616 |
| 3.250 | 0.583 | 0.549 | 0.701 | 0.615 |
| 3.500 | 0.561 | 0.552 | 0.688 | 0.585 |
| 3.750 | 0.560 | 0.552 | 0.682 | 0.576 |
| 4.000 | 0.578 | 0.548 | 0.675 | 0.559 |
