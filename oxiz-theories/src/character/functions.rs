//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)


/// Character value (Unicode code point)
pub type CodePoint = u32;
#[cfg(test)]
mod tests {
    use super::*;
    use crate::character::*;
    #[test]
    fn test_char_solver_creation() {
        let solver = CharSolver::new();
        assert_eq!(solver.num_vars(), 0);
        assert_eq!(solver.num_constraints(), 0);
    }
    #[test]
    fn test_char_var_creation() {
        let mut solver = CharSolver::new();
        let v1 = solver.new_var();
        let v2 = solver.new_var();
        assert_ne!(v1, v2);
        assert_eq!(solver.num_vars(), 2);
    }
    #[test]
    fn test_char_value() {
        let known = CharValue::Known(65);
        assert!(known.is_known());
        assert_eq!(known.code_point(), Some(65));
        assert_eq!(known.to_char(), Some('A'));
        let unknown = CharValue::Unknown;
        assert!(! unknown.is_known());
        assert_eq!(unknown.code_point(), None);
    }
    #[test]
    fn test_equality_constraint() {
        let mut solver = CharSolver::new();
        let v = solver.new_var();
        solver.add_constraint(v, CharConstraint::Eq(65));
        let result = solver.check();
        assert_eq!(result, CharResult::Sat);
        assert_eq!(solver.get_value(v), Some(& CharValue::Known(65)));
    }
    #[test]
    fn test_digit_constraint() {
        let mut solver = CharSolver::new();
        let v = solver.new_var();
        solver.add_constraint(v, CharConstraint::IsDigit);
        let result = solver.check();
        assert_eq!(result, CharResult::Sat);
        if let Some(CharValue::Known(cp)) = solver.get_value(v) {
            assert!(char::from_u32(* cp).unwrap().is_ascii_digit());
        }
    }
    #[test]
    fn test_range_constraint() {
        let mut solver = CharSolver::new();
        let v = solver.new_var();
        solver.add_constraint(v, CharConstraint::InRange(65, 90));
        let result = solver.check();
        assert_eq!(result, CharResult::Sat);
        if let Some(CharValue::Known(cp)) = solver.get_value(v) {
            assert!(* cp >= 65 && * cp <= 90);
        }
    }
    #[test]
    fn test_conflicting_constraints() {
        let mut solver = CharSolver::new();
        let v = solver.new_var();
        solver.add_constraint(v, CharConstraint::IsDigit);
        solver.add_constraint(v, CharConstraint::IsLetter);
        let result = solver.check();
        assert_eq!(result, CharResult::Unsat);
    }
    #[test]
    fn test_var_equality() {
        let mut solver = CharSolver::new();
        let v1 = solver.new_var();
        let v2 = solver.new_var();
        solver.add_constraint(v1, CharConstraint::Eq(65));
        solver.add_constraint(v2, CharConstraint::EqVar(v1));
        let result = solver.check();
        assert_eq!(result, CharResult::Sat);
        let val1 = solver.get_value(v1);
        let val2 = solver.get_value(v2);
        assert_eq!(val1, val2);
    }
    #[test]
    fn test_uppercase_lowercase() {
        assert_eq!(CharSolver::to_uppercase(97), Some(65));
        assert_eq!(CharSolver::to_lowercase(65), Some(97));
    }
    #[test]
    fn test_unicode_category() {
        assert_eq!(UnicodeCategory::from_code_point(65), UnicodeCategory::Letter);
        assert_eq!(UnicodeCategory::from_code_point(48), UnicodeCategory::Number);
        assert_eq!(UnicodeCategory::from_code_point(32), UnicodeCategory::Separator);
    }
    #[test]
    fn test_case_insensitive_eq() {
        assert!(CharSolver::case_insensitive_eq(65, 97));
        assert!(CharSolver::case_insensitive_eq(97, 65));
        assert!(! CharSolver::case_insensitive_eq(65, 66));
    }
    #[test]
    fn test_is_valid_code_point() {
        assert!(CharSolver::is_valid_code_point(65));
        assert!(CharSolver::is_valid_code_point(0x1F600));
        assert!(! CharSolver::is_valid_code_point(0xD800));
    }
    #[test]
    fn test_reset() {
        let mut solver = CharSolver::new();
        let v = solver.new_var();
        solver.add_constraint(v, CharConstraint::Eq(65));
        let _ = solver.check();
        assert!(solver.get_value(v).is_some());
        solver.reset();
        assert!(solver.get_value(v).is_none());
    }
    #[test]
    fn test_clear() {
        let mut solver = CharSolver::new();
        let _ = solver.new_var();
        solver.add_constraint(CharVar::new(0), CharConstraint::Eq(65));
        solver.clear();
        assert_eq!(solver.num_vars(), 0);
        assert_eq!(solver.num_constraints(), 0);
    }
    #[test]
    fn test_ascii_constraint() {
        let mut solver = CharSolver::new();
        let v = solver.new_var();
        solver.add_constraint(v, CharConstraint::IsAscii);
        let result = solver.check();
        assert_eq!(result, CharResult::Sat);
        if let Some(CharValue::Known(cp)) = solver.get_value(v) {
            assert!(* cp <= 127);
        }
    }
    #[test]
    fn test_stats() {
        let mut solver = CharSolver::new();
        let v = solver.new_var();
        solver.add_constraint(v, CharConstraint::Eq(65));
        let _ = solver.check();
        let stats = solver.stats();
        assert!(stats.constraints_added > 0);
    }
    #[test]
    fn test_unicode_block_basic_latin() {
        assert_eq!(UnicodeBlock::from_code_point(65), UnicodeBlock::BasicLatin);
        assert_eq!(UnicodeBlock::from_code_point(0), UnicodeBlock::BasicLatin);
        assert_eq!(UnicodeBlock::from_code_point(127), UnicodeBlock::BasicLatin);
    }
    #[test]
    fn test_unicode_block_cjk() {
        assert_eq!(
            UnicodeBlock::from_code_point(0x4E00), UnicodeBlock::CjkUnifiedIdeographs
        );
        assert_eq!(
            UnicodeBlock::from_code_point(0x9FFF), UnicodeBlock::CjkUnifiedIdeographs
        );
    }
    #[test]
    fn test_unicode_block_range() {
        let block = UnicodeBlock::BasicLatin;
        let (low, high) = block.range();
        assert_eq!(low, 0x0000);
        assert_eq!(high, 0x007F);
        assert!(block.contains(65));
        assert!(! block.contains(128));
    }
    #[test]
    fn test_unicode_script_latin() {
        assert_eq!(UnicodeScript::from_code_point(65), UnicodeScript::Latin);
        assert_eq!(UnicodeScript::from_code_point(97), UnicodeScript::Latin);
    }
    #[test]
    fn test_unicode_script_cjk() {
        assert_eq!(UnicodeScript::from_code_point(0x4E00), UnicodeScript::Han);
    }
    #[test]
    fn test_unicode_script_direction() {
        assert!(UnicodeScript::Latin.is_ltr());
        assert!(! UnicodeScript::Arabic.is_ltr());
        assert!(! UnicodeScript::Hebrew.is_ltr());
    }
    #[test]
    fn test_unicode_script_east_asian() {
        assert!(UnicodeScript::Han.is_east_asian());
        assert!(UnicodeScript::Hiragana.is_east_asian());
        assert!(! UnicodeScript::Latin.is_east_asian());
    }
    #[test]
    fn test_char_width_narrow() {
        assert_eq!(CharWidth::from_code_point(65), CharWidth::Narrow);
        assert_eq!(CharWidth::Narrow.cells(), 1);
    }
    #[test]
    fn test_char_width_wide() {
        assert_eq!(CharWidth::from_code_point(0x4E00), CharWidth::Wide);
        assert_eq!(CharWidth::Wide.cells(), 2);
    }
    #[test]
    fn test_char_width_fullwidth() {
        assert_eq!(CharWidth::from_code_point(0xFF21), CharWidth::Fullwidth);
        assert_eq!(CharWidth::Fullwidth.cells(), 2);
    }
    #[test]
    fn test_char_class_digit() {
        let class = CharClass::Digit;
        assert!(class.matches(48));
        assert!(class.matches(57));
        assert!(! class.matches(65));
    }
    #[test]
    fn test_char_class_alpha() {
        let class = CharClass::Alpha;
        assert!(class.matches(65));
        assert!(class.matches(97));
        assert!(! class.matches(48));
    }
    #[test]
    fn test_char_class_range() {
        let class = CharClass::Range(65, 90);
        assert!(class.matches(65));
        assert!(class.matches(90));
        assert!(! class.matches(91));
    }
    #[test]
    fn test_char_class_negation() {
        let class = CharClass::Negation(Box::new(CharClass::Digit));
        assert!(! class.matches(48));
        assert!(class.matches(65));
    }
    #[test]
    fn test_char_class_union() {
        let class = CharClass::Union(
            vec![CharClass::Range(65, 90), CharClass::Range(97, 122)],
        );
        assert!(class.matches(65));
        assert!(class.matches(97));
        assert!(! class.matches(48));
    }
    #[test]
    fn test_char_class_from_pattern() {
        assert!(CharClass::from_pattern(r"\d").is_some());
        assert!(CharClass::from_pattern(r"\w").is_some());
        assert!(CharClass::from_pattern(".").is_some());
        assert!(CharClass::from_pattern("invalid").is_none());
    }
    #[test]
    fn test_char_class_enumerate_ascii() {
        let class = CharClass::Digit;
        let digits = class.enumerate_ascii();
        assert_eq!(digits.len(), 10);
        assert_eq!(digits[0], 48);
        assert_eq!(digits[9], 57);
    }
    #[test]
    fn test_char_domain_full() {
        let domain = CharDomain::full();
        assert!(! domain.is_empty());
        assert!(domain.contains(0));
        assert!(domain.contains(65));
        assert!(domain.contains(0x10FFFF));
    }
    #[test]
    fn test_char_domain_empty() {
        let domain = CharDomain::empty();
        assert!(domain.is_empty());
        assert!(! domain.contains(65));
    }
    #[test]
    fn test_char_domain_singleton() {
        let domain = CharDomain::singleton(65);
        assert_eq!(domain.is_singleton(), Some(65));
        assert!(domain.contains(65));
        assert!(! domain.contains(66));
    }
    #[test]
    fn test_char_domain_range() {
        let domain = CharDomain::range(65, 90);
        assert!(domain.contains(65));
        assert!(domain.contains(90));
        assert!(! domain.contains(64));
        assert!(! domain.contains(91));
    }
    #[test]
    fn test_char_domain_intersect() {
        let mut d1 = CharDomain::range(60, 80);
        let d2 = CharDomain::range(70, 90);
        d1.intersect(&d2);
        assert!(d1.contains(70));
        assert!(d1.contains(80));
        assert!(! d1.contains(60));
        assert!(! d1.contains(90));
    }
    #[test]
    fn test_char_domain_exclude() {
        let mut domain = CharDomain::range(65, 67);
        domain.exclude(66);
        assert!(domain.contains(65));
        assert!(! domain.contains(66));
        assert!(domain.contains(67));
    }
    #[test]
    fn test_char_domain_min_max() {
        let domain = CharDomain::range(65, 90);
        assert_eq!(domain.min(), Some(65));
        assert_eq!(domain.max(), Some(90));
    }
    #[test]
    fn test_char_domain_size() {
        let domain = CharDomain::range(65, 74);
        assert_eq!(domain.size(), 10);
    }
    #[test]
    fn test_advanced_solver_creation() {
        let solver = AdvancedCharSolver::new();
        assert_eq!(solver.stats().constraints_added, 0);
    }
    #[test]
    fn test_advanced_solver_domain() {
        let mut solver = AdvancedCharSolver::new();
        let v = solver.new_var();
        let domain = solver.get_domain(v);
        assert!(domain.is_some());
        assert!(! domain.map(|d: &CharDomain| d.is_empty()).unwrap_or(true));
    }
    #[test]
    fn test_advanced_solver_constraint() {
        let mut solver = AdvancedCharSolver::new();
        let v = solver.new_var();
        solver.add_constraint(v, CharConstraint::IsDigit);
        let result = solver.check();
        assert_eq!(result, CharResult::Sat);
    }
    #[test]
    fn test_advanced_solver_conflicting() {
        let mut solver = AdvancedCharSolver::new();
        let v = solver.new_var();
        solver.add_constraint(v, CharConstraint::IsDigit);
        solver.add_constraint(v, CharConstraint::IsUppercase);
        let result = solver.check();
        assert_eq!(result, CharResult::Unsat);
    }
    #[test]
    fn test_normalizer_nfd() {
        let mut normalizer = CharNormalizer::new(NormalizationForm::Nfd);
        let decomposed = normalizer.normalize(&[0x00C0]);
        assert_eq!(decomposed, vec![0x0041, 0x0300]);
    }
    #[test]
    fn test_normalizer_nfc() {
        let mut normalizer = CharNormalizer::new(NormalizationForm::Nfc);
        let composed = normalizer.normalize(&[0x0041, 0x0300]);
        assert_eq!(composed, vec![0x00C0]);
    }
    #[test]
    fn test_normalizer_combining_class() {
        let mut normalizer = CharNormalizer::new(NormalizationForm::Nfd);
        assert_eq!(normalizer.combining_class(0x0300), 230);
        assert_eq!(normalizer.combining_class(65), 0);
    }
    #[test]
    fn test_normalizer_is_combining() {
        let normalizer = CharNormalizer::new(NormalizationForm::Nfd);
        assert!(normalizer.is_combining(0x0300));
        assert!(! normalizer.is_combining(65));
    }
    #[test]
    fn test_case_folder_simple() {
        let mut folder = CaseFolder::new(CaseFoldMode::Simple);
        assert_eq!(folder.fold(65), vec![97]);
        assert_eq!(folder.fold(97), vec![97]);
    }
    #[test]
    fn test_case_folder_full() {
        let mut folder = CaseFolder::new(CaseFoldMode::Full);
        assert_eq!(folder.fold(0x00DF), vec![0x0073, 0x0073]);
    }
    #[test]
    fn test_case_folder_turkic() {
        let mut folder = CaseFolder::new(CaseFoldMode::Turkic);
        assert_eq!(folder.fold(0x0049), vec![0x0131]);
        assert_eq!(folder.fold(0x0130), vec![0x0069]);
    }
    #[test]
    fn test_case_folder_equals() {
        let mut folder = CaseFolder::new(CaseFoldMode::Simple);
        assert!(folder.equals(& [65, 66, 67], & [97, 98, 99]));
        assert!(! folder.equals(& [65], & [66]));
    }
}
