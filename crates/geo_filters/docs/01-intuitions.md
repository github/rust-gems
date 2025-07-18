# Intuition

If you're not familiar with probabilistic data structures then getting an intuition for how this data structure works
using a real world example might be helpful.

Imagine you wanted to count how many raindrops fall from the sky and landing in certain area, let's say one square meter.
A single raindrop will fall randomly hitting the ground in the area you wish to analyze. Trying to count every single drop
that hits the ground would be very difficult, there are simply far too many.

Perhaps you can estimate the number of raindrops?

Instead of counting each and every drop of rain you could instead lay out a grid of buckets and then count how many buckets
have *any* raindrops in them at all. For this thought experiment we're not considering how _much_ water is in them, only if
there is a non-zero amount of water. Uniformly sized buckets might work ok for a small shower, but you'd quickly run
into an issue where most of your buckets have some amount rain in them. Because of this, you would not be able to differentiate between a gentle shower and a downpour; either way most of the buckets have _some_ water in them.

By varying the size of the buckets you reduce the probability that a raindrop will land in the smaller ones. You can
then estimate the number of droplets by adding up the probabilities that a given bucket has a raindrop in it. Smaller
ones are much less likely to have a droplet in so if you've got a lot of smaller buckets with drop lets in, that would imply
that there was a lot of rain. If those buckets are mostly dry, then it would imply that there was only a small amount
of drizzle. You still need a wide range of bucket sizes to be able to tell the difference between having no rain and a small
amount of rain.

You can estimate the difference in the amount of rain fall on two areas by counting the number of buckets where the matching
bucket size has rain in it in one area but not the other.

This data structure works in a similar way. Items are hashed to produce a "random" number which we assign to a bucket. The
bucket "sizes" are arranged to follow a geometric distribution to allow us to calculate an estimate of the number of items
using well known formulas.