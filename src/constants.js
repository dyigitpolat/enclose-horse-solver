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

  // Default solve time budget (users can override in the UI).
  HP.TIME_BUDGET_MS = 10_000;
  // Safety cap for the UI/worker time budget to avoid accidental multi-minute solves.
  HP.TIME_BUDGET_MS_MAX = 120_000;
})();

