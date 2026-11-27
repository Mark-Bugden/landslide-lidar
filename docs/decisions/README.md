# Decision records

One file per task, `docs/decisions/<task-id>.md` (for example `P1-08.md`). Record every interpretation of the
method, every choice the brief leaves open and every deviation. P1-16 folds them all into `PREREGISTRATION.md` at the
freeze; after that, nothing here changes the method.

Each decision is one section:

```markdown
## D08 · 15 km block grid            <- register id from PLAN.md §6, or "Dxx-new" if not in the register

- **Decision.** What we do, precisely enough to re-implement.
- **Why.** The reason, citing BRIEF.md (section) or PLAN.md where it applies.
- **Alternatives.** The other readings considered, and why not.
- **Evidence.** Counts or checks, e.g. "18,649 in_mapped polygons [E], spike 18,649". Tag claims
  [V] verified, [E] empirical (our run) or [I] inference.
- **Deviation?** No, or Yes: how it departs from Appendix A or B and why (deviations are allowed only before training).
```

If a §5 reference number was used to choose between readings of ambiguous text, say so under **Evidence**.
