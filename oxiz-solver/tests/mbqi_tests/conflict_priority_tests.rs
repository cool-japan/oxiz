use oxiz_core::ast::TermId;
use oxiz_solver::mbqi::ConflictScores;

#[test]
fn conflict_score_increments_on_conflict() {
    let mut scores = ConflictScores::new(0.5);
    let qid = TermId::new(7);

    scores.record_conflict(qid);
    scores.record_conflict(qid);

    assert_eq!(scores.score(qid), Some(2));
    assert_eq!(scores.priority_order(), vec![qid]);
}

#[test]
fn decay_on_restart_reduces_scores() {
    let mut scores = ConflictScores::new(0.5);
    let qid = TermId::new(3);

    scores.record_conflict(qid);
    scores.record_conflict(qid);
    scores.decay_on_restart();

    assert_eq!(scores.score(qid), Some(1));
}
