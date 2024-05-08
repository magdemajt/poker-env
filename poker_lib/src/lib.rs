use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;
use rs_poker::core::{Card, Deck, Hand, Rank, Rankable};

fn get_cards_from_string_vec(string_vec: Vec<String>) -> Vec<Card> {
    string_vec
        .iter()
        .map(|arg| {
            Card::try_from(&arg[..])
                .expect("Invalid args string. Expected a string of 2 characters.")
        })
        .collect()
}

fn play_random_game(number_of_players: u32, given_cards: &Vec<Card>) -> bool {
    let (main_agent_cards, given_table_cards) = given_cards.split_at(2);
    let mut shuffled_deck_tail = Deck::default()
        .iter()
        .filter(|x| !main_agent_cards.contains(x) && !given_table_cards.contains(x))
        .cloned()
        .collect::<Vec<Card>>();
    shuffled_deck_tail.shuffle(&mut thread_rng());

    let missing_table_cards_count = 5 - given_table_cards.len();

    let (missing_table_cards, shuffled_deck_tail) =
        shuffled_deck_tail.split_at(missing_table_cards_count);

    let all_table_cards = given_table_cards
        .iter()
        .chain(missing_table_cards)
        .cloned()
        .collect::<Vec<Card>>();

    let other_ranks: Vec<Rank> = (0..number_of_players as usize - 1)
        .map(|i| {
            let player_cards: Vec<Card> = all_table_cards
                .iter()
                .chain(&shuffled_deck_tail[i * 2..i * 2 + 2])
                .cloned()
                .collect();
            Hand::new_with_cards(player_cards).rank()
        })
        .collect();

    let agent_rank = Hand::new_with_cards(
        main_agent_cards
            .iter()
            .chain(all_table_cards.iter())
            .cloned()
            .collect(),
    )
    .rank();

    let game_won: bool = other_ranks.iter().all(|r| r < &agent_rank);

    return game_won;
}

#[pyfunction]
fn get_chances(cards: Vec<String>, num_players: u32, iterations: u32) -> f32 {
    let cards = get_cards_from_string_vec(cards);

    if num_players < 2 {
        panic!("Invalid number of players. Expected at least 2.");
    }
    let allowed_card_counts = vec![2, 5, 6, 7];
    let card_count = cards.len();
    if !allowed_card_counts.contains(&card_count) {
        panic!("Invalid number of cards. Expected 2, 5, 6, or 7.");
    }

    let card_set = cards.iter().collect::<std::collections::HashSet<_>>();
    if card_set.len() != card_count {
        panic!("Duplicate cards detected.");
    }

    let total_wins = (0..iterations)
        .into_par_iter()
        .map(|_| play_random_game(num_players, &cards))
        .filter(|&x| x)
        .count() as f32;

    return total_wins / iterations as f32;
}

#[pyfunction]
fn get_win_indices(table_cards: Vec<String>, players_hands: Vec<Vec<String>>) -> Vec<u8> {
    let table_cards = get_cards_from_string_vec(table_cards);
    let players_hands = players_hands
        .iter()
        .map(|x| get_cards_from_string_vec(x.clone()))
        .collect::<Vec<Vec<Card>>>();

    let extended_players_hands = players_hands
        .iter()
        .map(|x| {
            table_cards
                .iter()
                .chain(x.iter())
                .cloned()
                .collect::<Vec<Card>>()
        })
        .collect::<Vec<Vec<Card>>>();

    let ranks = extended_players_hands
        .iter()
        .map(|x| Hand::new_with_cards(x.clone()).rank())
        .collect::<Vec<Rank>>();

    let max_rank = ranks.iter().max().unwrap();
    return ranks
        .iter()
        .enumerate()
        .filter(|(_, x)| *x == max_rank)
        .map(|(i, _)| i as u8)
        .collect::<Vec<u8>>();
}

#[pymodule]
fn poker_lib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_chances, m)?)?;
    m.add_function(wrap_pyfunction!(get_win_indices, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::get_chances;
    use crate::get_win_indices;

    #[test]
    fn exploration() {
        let chances = get_chances(vec!["As".to_string(), "Ah".to_string()], 2, 100000);
        print!("{:?}", chances)
    }

    #[test]
    fn winner_index() {
        let winner_index = get_win_indices(
            vec!["As".to_string(), "Ah".to_string(), "Ad".to_string()],
            vec![
                vec!["Ks".to_string(), "Kh".to_string()],
                vec!["Qs".to_string(), "Qh".to_string()],
            ],
        );
        print!("{:?}", winner_index)
    }
}
