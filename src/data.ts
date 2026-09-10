import type { Movie, Profile } from './lib/recommendation'

export const catalog: Movie[] = [
  { id: 'aftersun-2022', title: 'Aftersun', year: 2022, runtime: 102, genres: ['Drama'], platforms: ['Filmin', 'Prime Video'], rating: 4.2, tone: 'Tender', director: 'Charlotte Wells', language: 'English', moods: ['Gentle', 'Reflective'], poster: 'https://image.tmdb.org/t/p/w500/jeXmhP2zbUkREMRqFOYIwQOk49T.jpg' },
  { id: 'clue-1985', title: 'Clue', year: 1985, runtime: 94, genres: ['Comedy', 'Mystery'], platforms: ['Movistar Plus+', 'SkyShowtime'], rating: 3.8, tone: 'Playful', director: 'Jonathan Lynn', language: 'English', moods: ['Playful'], poster: 'https://image.tmdb.org/t/p/w500/aRxbYOYHS8T73nzR8hsLousoplR.jpg' },
  { id: 'decision-2022', title: 'Decision to Leave', year: 2022, runtime: 138, genres: ['Romance', 'Mystery'], platforms: ['MUBI', 'Filmin'], rating: 4.0, tone: 'Lush', director: 'Park Chan-wook', language: 'Korean', moods: ['Reflective', 'Twisty'], poster: 'https://image.tmdb.org/t/p/w500/N0rskxERWcFbuZQ1DpQq2FQ4Xa.jpg' },
  { id: 'perfect-days-2023', title: 'Perfect Days', year: 2023, runtime: 124, genres: ['Drama'], platforms: ['Movistar Plus+', 'Filmin'], rating: 4.2, tone: 'Gentle', director: 'Wim Wenders', language: 'Japanese', moods: ['Gentle', 'Reflective'], poster: 'https://image.tmdb.org/t/p/w500/mjEk5Wwx6TYVqw29zSaUHclMIgp.jpg' },
  { id: 'rye-lane-2023', title: 'Rye Lane', year: 2023, runtime: 82, genres: ['Comedy', 'Romance'], platforms: ['Disney+'], rating: 3.7, tone: 'Bright', director: 'Raine Allen-Miller', language: 'English', moods: ['Playful'], poster: 'https://placehold.co/300x450/e95b37/ffffff?text=RYE%0ALANE' },
  { id: 'coherence-2013', title: 'Coherence', year: 2013, runtime: 89, genres: ['Mystery', 'Sci-Fi'], platforms: ['Filmin', 'Prime Video'], rating: 3.8, tone: 'Twisty', director: 'James Ward Byrkit', language: 'English', moods: ['Twisty'], poster: 'https://image.tmdb.org/t/p/w500/ezUtb9m5DeLwL2gxi4gktzNCvQv.jpg' },
  { id: 'past-lives-2023', title: 'Past Lives', year: 2023, runtime: 106, genres: ['Drama', 'Romance'], platforms: ['Movistar Plus+'], rating: 4.1, tone: 'Reflective', director: 'Celine Song', language: 'English', moods: ['Gentle', 'Reflective'], poster: 'https://image.tmdb.org/t/p/w500/k3waqVXSnvCZWfJYNtdamTgTtTA.jpg' },
  { id: 'attack-block-2011', title: 'Attack the Block', year: 2011, runtime: 88, genres: ['Comedy', 'Sci-Fi'], platforms: ['Netflix', 'Prime Video'], rating: 3.5, tone: 'Electric', director: 'Joe Cornish', language: 'English', moods: ['Electric', 'Playful'], poster: 'https://image.tmdb.org/t/p/w500/vVQwgfS9gSFviVT4gS7tZAmhRFc.jpg' },
]

export const demoProfiles: Profile[] = [
  { id: 'maya', name: 'Maya', color: '#f36b35', ratings: [{ title: 'Moonlight', year: 2016, rating: 5 }], watched: [], preferredGenres: ['Drama', 'Romance', 'Sci-Fi'] },
  { id: 'leo', name: 'Leo', color: '#11a683', ratings: [{ title: 'Knives Out', year: 2019, rating: 4.5 }], watched: [], preferredGenres: ['Comedy', 'Mystery', 'Sci-Fi'] },
]

export const profileColors = ['#e04d2d', '#008f72', '#4169c1', '#d39a18', '#9b5d9c']