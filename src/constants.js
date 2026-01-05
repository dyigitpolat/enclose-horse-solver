(() => {
  const HP = (window.HorsePen = window.HorsePen || {});

  HP.THRESHOLDS = Object.freeze({
    WATER_BLUE: 50,
    HORSE_BRIGHTNESS: 160,
  });

  // Scoring: each enclosed tile is worth 1. Special items can add bonus points.
  HP.SCORING = Object.freeze({
    CHERRY_BONUS: 3,
  });

  HP.TIME_BUDGET_MS = 10_000;
})();

