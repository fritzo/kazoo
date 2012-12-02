
#include "rational.h"

void test_ball ()
{
  auto ball = Rational::ball_of_radius(4);
  ASSERT_EQ(ball.size(), 7);
  ASSERT_EQ(ball[0], Rational::Number(1, 3));
  ASSERT_EQ(ball[1], Rational::Number(1, 2));
  ASSERT_EQ(ball[2], Rational::Number(2, 3));
  ASSERT_EQ(ball[3], Rational::Number(1, 1));
  ASSERT_EQ(ball[4], Rational::Number(3, 2));
  ASSERT_EQ(ball[5], Rational::Number(2, 1));
  ASSERT_EQ(ball[6], Rational::Number(3, 1));
}

int main ()
{
  test_ball();

  return 0;
}

